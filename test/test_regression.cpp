// Regression tests: cross-backend replication with deduplication, durability
// after move(), reclaimed blobs, id allocation after recovery, and concurrent
// publish/read and eviction on one runtime.
#include "test_util.h"
#include "storagefabric/core/runtime.h"
#include "storagefabric/storage/synthetic_backend.h"

#include <atomic>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

using namespace storagefabric;

static int g_fail = 0;
#define EXPECT(cond)                                                                  \
    do {                                                                              \
        if (!(cond)) { ++g_fail; std::fprintf(stderr, "[FAILED] %s (line %d)\n", #cond, __LINE__); } \
    } while (0)

namespace {

std::filesystem::path fresh_dir(const std::string& leaf) {
    const auto d = std::filesystem::temp_directory_path() / ("sfb-test-regression-" + leaf);
    std::error_code ec;
    std::filesystem::remove_all(d, ec);
    return d;
}

AuthorityEnvelope runner_authority() {
    AuthorityEnvelope a;
    a.epoch = CoordinatorEpoch(1);
    a.boot = WorkerBootId(100);
    a.worker = WorkerId(7);
    a.generation = AuthorityGeneration(1);
    a.origin = AuthorityOrigin::RUNNER;
    return a;
}

std::vector<std::uint8_t> payload(std::size_t n, unsigned seed) {
    std::vector<std::uint8_t> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = static_cast<std::uint8_t>((i * 31 + seed) & 0xFF);
    return v;
}

struct Rig {
    StorageFabric sf;
    StorageBackendId a, b, c;
    explicit Rig(const std::string& tag) {
        sf.set_authority(runner_authority());
        a = sf.register_local_backend("a", fresh_dir(tag + "-a")).value();
        b = sf.register_local_backend("b", fresh_dir(tag + "-b")).value();
        c = sf.register_local_backend("c", fresh_dir(tag + "-c")).value();
    }
    ObjectDescriptor define(const std::vector<std::uint8_t>& data) {
        DurabilityRequirement dur; dur.min_replicas = 1;
        return sf.define_object(ObjectKind::CHECKPOINT, data.size(), ByteSpan(data.data(), data.size()),
                                runner_authority(), dur).value();
    }
};

bool reads_back(StorageFabric& sf, ObjectId id, const std::vector<std::uint8_t>& data,
                std::optional<PlacementId> p = std::nullopt) {
    auto r = sf.read(id, p);
    return r.ok() && r.value() == data;
}

// replicate()/move() with default options (dedupe on) must store the content on
// the target backend: a replica is read from its own backend.
void test_replicate_and_move_store_on_target() {
    Rig rig("dedup");
    const auto data = payload(4096, 3);
    const ByteSpan span(data.data(), data.size());
    const auto obj = rig.define(data);
    PublishOptions opts;
    auto p1 = rig.sf.publish_to(obj, span, rig.a, opts);
    EXPECT(p1.ok());
    if (!p1.ok()) return;
    auto p2 = rig.sf.replicate(p1.value(), rig.b, opts);
    EXPECT(p2.ok());
    if (p2.ok()) EXPECT(reads_back(rig.sf, obj.id, data, p2.value().id));

    const auto data2 = payload(4096, 9);
    const auto obj2 = rig.define(data2);
    auto q1 = rig.sf.publish_to(obj2, ByteSpan(data2.data(), data2.size()), rig.a, opts);
    EXPECT(q1.ok());
    if (!q1.ok()) return;
    auto q2 = rig.sf.move(q1.value(), rig.b, opts);
    EXPECT(q2.ok());
    EXPECT(reads_back(rig.sf, obj2.id, data2));
}

// After move() the source copy is stale: the moved-to copy is the only
// authoritative one and cannot be evicted.
void test_move_keeps_last_copy_protected() {
    Rig rig("move");
    const auto data = payload(2048, 5);
    const auto obj = rig.define(data);
    PublishOptions opts; opts.dedupe = false;
    auto p1 = rig.sf.publish_to(obj, ByteSpan(data.data(), data.size()), rig.a, opts);
    EXPECT(p1.ok());
    if (!p1.ok()) return;
    auto p2 = rig.sf.move(p1.value(), rig.b, opts);
    EXPECT(p2.ok());
    if (!p2.ok()) return;
    EXPECT(rig.sf.can_evict(p2.value().id).code == StatusCode::EvictionUnsafe);
    auto ev = rig.sf.evict(p2.value().id);
    EXPECT(ev.ok() && !ev.value().allowed);
    EXPECT(reads_back(rig.sf, obj.id, data));

    // a -> b -> c: after evicting the stale a copy, c is still protected.
    auto p3 = rig.sf.move(p2.value(), rig.c, opts);
    EXPECT(p3.ok());
    if (!p3.ok()) return;
    auto ev_a = rig.sf.evict(p1.value().id);
    EXPECT(ev_a.ok() && ev_a.value().allowed);
    EXPECT(rig.sf.can_evict(p3.value().id).code == StatusCode::EvictionUnsafe);
    EXPECT(reads_back(rig.sf, obj.id, data));
}

// A blob reclaimed by eviction leaves the dedup index: publishing the same
// content again stores it instead of referencing the deleted blob.
void test_reclaimed_blob_not_reused() {
    Rig rig("reclaim");
    const auto data = payload(1024, 11);
    const auto obj = rig.define(data);
    PublishOptions nodedup; nodedup.dedupe = false;
    auto pa = rig.sf.publish_to(obj, ByteSpan(data.data(), data.size()), rig.a, nodedup);
    EXPECT(pa.ok());
    if (!pa.ok()) return;
    auto pb = rig.sf.replicate(pa.value(), rig.b, nodedup);
    EXPECT(pb.ok());
    if (!pb.ok()) return;
    auto ev = rig.sf.evict(pb.value().id);      // two authoritative copies: allowed
    EXPECT(ev.ok() && ev.value().allowed);
    const auto obj2 = rig.define(data);
    PublishOptions dedup;
    auto pz = rig.sf.publish_to(obj2, ByteSpan(data.data(), data.size()), rig.b, dedup);
    EXPECT(pz.ok());
    EXPECT(reads_back(rig.sf, obj2.id, data));
}

// Objects and publications after recover() get ids past the recovered ones.
void test_ids_after_recover() {
    const auto dir = fresh_dir("recover");
    const auto meta = std::filesystem::temp_directory_path() / "sfb-test-regression-recover.meta";
    const auto data = payload(512, 2);
    {
        StorageFabric sf;
        sf.set_authority(runner_authority());
        auto bk = sf.register_local_backend("a", dir);
        EXPECT(bk.ok());
        DurabilityRequirement dur; dur.min_replicas = 1;
        auto obj = sf.define_object(ObjectKind::CHECKPOINT, data.size(), ByteSpan(data.data(), data.size()), runner_authority(), dur);
        EXPECT(obj.ok());
        if (obj.ok()) EXPECT(sf.publish_to(obj.value(), ByteSpan(data.data(), data.size()), bk.value(), PublishOptions{}).ok());
        EXPECT(sf.save(meta).ok());
    }
    StorageFabric sf;
    auto bk = sf.register_local_backend("a", dir);
    EXPECT(bk.ok());
    EXPECT(sf.recover(meta).ok());
    const auto data2 = payload(512, 8);
    DurabilityRequirement dur; dur.min_replicas = 1;
    auto obj = sf.define_object(ObjectKind::CHECKPOINT, data2.size(), ByteSpan(data2.data(), data2.size()), sf.authority(), dur);
    EXPECT(obj.ok());
    if (obj.ok()) {
        auto p = sf.publish_to(obj.value(), ByteSpan(data2.data(), data2.size()), bk.value(), PublishOptions{});
        EXPECT(p.ok());
        EXPECT(reads_back(sf, obj.value().id, data2));
    }
    EXPECT(sf.object_count() == 2);
    std::error_code ec;
    std::filesystem::remove(meta, ec);
}

// Publishing while another thread reads and lists placements (the coordinator
// serves CREATE_OBJECT and READ_OBJECT on different connections).
void test_publish_concurrent_with_reads() {
    StorageFabric sf;
    sf.set_authority(runner_authority());
    SyntheticProfile prof;
    auto bk = sf.register_synthetic_backend("syn", prof);
    EXPECT(bk.ok());
    if (!bk.ok()) return;
    DurabilityRequirement dur; dur.min_replicas = 1;
    const auto first = payload(64, 1);
    auto o0 = sf.define_object(ObjectKind::CHECKPOINT, first.size(), ByteSpan(first.data(), first.size()), runner_authority(), dur);
    EXPECT(o0.ok() && sf.publish_to(o0.value(), ByteSpan(first.data(), first.size()), bk.value(), PublishOptions{}).ok());
    std::atomic<bool> done{false};
    std::atomic<int> bad_reads{0};
    std::thread reader([&] {
        while (!done.load()) {
            if (!reads_back(sf, o0.value().id, first)) bad_reads.fetch_add(1);
            (void)sf.placements();
        }
    });
    for (unsigned i = 0; i < 3000; ++i) {
        const auto d = payload(64, 100 + i);
        auto o = sf.define_object(ObjectKind::CHECKPOINT, d.size(), ByteSpan(d.data(), d.size()), runner_authority(), dur);
        if (!o.ok() || !sf.publish_to(o.value(), ByteSpan(d.data(), d.size()), bk.value(), PublishOptions{}).ok()) bad_reads.fetch_add(1);
    }
    done.store(true);
    reader.join();
    EXPECT(bad_reads.load() == 0);
    EXPECT(sf.placement_count() == 3001);
}

// Two concurrent evictions of one placement release its blob once: an object
// sharing that blob stays readable.
void test_concurrent_evictions_release_once() {
    for (int iter = 0; iter < 40; ++iter) {
        Rig rig("evict");
        const auto data = payload(1024, 21);
        const auto x = rig.define(data);
        PublishOptions nodedup; nodedup.dedupe = false;
        auto pxa = rig.sf.publish_to(x, ByteSpan(data.data(), data.size()), rig.a, nodedup);
        if (!pxa.ok()) { EXPECT(pxa.ok()); return; }
        const auto y = rig.define(data);
        EXPECT(rig.sf.publish_to(y, ByteSpan(data.data(), data.size()), rig.a, PublishOptions{}).ok());  // shares x's blob on a
        EXPECT(rig.sf.replicate(pxa.value(), rig.b, nodedup).ok());   // x has a second copy: evicting pxa is allowed
        std::atomic<int> ready{0};
        auto evict = [&] {
            ready.fetch_add(1);
            while (ready.load() < 2) {}
            (void)rig.sf.evict(pxa.value().id);
        };
        std::thread t1(evict), t2(evict);
        t1.join(); t2.join();
        if (!reads_back(rig.sf, y.id, data)) { EXPECT(reads_back(rig.sf, y.id, data)); return; }
    }
}

}  // namespace

int main() {
    std::printf("test_regression starting\n");
    test_replicate_and_move_store_on_target();
    test_move_keeps_last_copy_protected();
    test_reclaimed_blob_not_reused();
    test_ids_after_recover();
    test_publish_concurrent_with_reads();
    test_concurrent_evictions_release_once();
    std::printf("test_regression %s (failures=%d)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
