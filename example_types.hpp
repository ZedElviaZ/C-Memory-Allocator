#pragma once

// -----------------------------------------------------------------------------
// example_types.hpp
//
// Representative object types used in tests and examples.
// Each type represents a realistic use case for a fixed-size pool:
//
//   Bullet       — game engine projectile. High churn: many created/destroyed
//                  per frame. Typical pool size: 256–4096.
//
//   NetworkPacket — network server payload. Created on receipt, destroyed
//                   after processing. Typical pool size: 1024–65536.
//
//   Particle     — visual effect particle. Extremely high churn, short life.
//                  Typical pool size: 4096–65536.
//
//   GameEntity   — scene entity. Lower churn, longer life. Pool prevents
//                  fragmentation over a long session.
//
// All types satisfy the FixedPoolManager constraints:
//   sizeof(T) >= sizeof(void*)     — all are larger than 8 bytes
//   noexcept destructor            — all destructors are noexcept
// -----------------------------------------------------------------------------

#include <array>
#include <cstdint>
#include <cstring>

namespace fp::examples {

// ---------------------------------------------------------------------------
// Bullet
//
// Minimal game projectile. Fits in one cache line (64 bytes).
// ---------------------------------------------------------------------------
struct Bullet {
    float    x, y, z;          // position (12 bytes)
    float    vx, vy, vz;       // velocity (12 bytes)
    float    damage;            // (4 bytes)
    uint32_t owner_id;          // (4 bytes)
    uint32_t flags;             // (4 bytes)
    float    lifetime_remaining;// (4 bytes)
    // Total: 44 bytes. sizeof >= sizeof(void*). ✓

    Bullet() noexcept
        : x{0}, y{0}, z{0}
        , vx{0}, vy{0}, vz{0}
        , damage{0}
        , owner_id{0}
        , flags{0}
        , lifetime_remaining{0}
    {}

    Bullet(float px, float py, float pz,
           float pvx, float pvy, float pvz,
           float dmg, uint32_t owner) noexcept
        : x{px}, y{py}, z{pz}
        , vx{pvx}, vy{pvy}, vz{pvz}
        , damage{dmg}
        , owner_id{owner}
        , flags{0}
        , lifetime_remaining{5.0f}
    {}

    ~Bullet() noexcept = default;
};

// ---------------------------------------------------------------------------
// NetworkPacket
//
// Fixed-size packet buffer. The pool avoids per-packet heap allocation
// in tight receive loops.
// ---------------------------------------------------------------------------
struct NetworkPacket {
    static constexpr std::size_t kMaxPayload = 1400;  // typical MTU payload

    uint32_t source_ip;
    uint32_t dest_ip;
    uint16_t source_port;
    uint16_t dest_port;
    uint32_t sequence_num;
    uint16_t payload_len;
    uint16_t flags;
    std::array<uint8_t, kMaxPayload> payload;

    NetworkPacket() noexcept
        : source_ip{0}, dest_ip{0}
        , source_port{0}, dest_port{0}
        , sequence_num{0}
        , payload_len{0}
        , flags{0}
        , payload{}
    {}

    ~NetworkPacket() noexcept = default;
};

static_assert(sizeof(NetworkPacket) >= sizeof(void*),
    "NetworkPacket too small for pool");

// ---------------------------------------------------------------------------
// Particle
//
// Visual effect particle. Smallest of the example types.
// Note: if you need sizeof(Particle) < sizeof(void*), use the external
// index strategy and DO NOT instantiate FixedPoolManager<Particle, ...>
// with the intrusive list.
// ---------------------------------------------------------------------------
struct Particle {
    float x, y, z;
    float r, g, b, a;   // colour
    float size;
    float age;
    float max_age;

    Particle() noexcept
        : x{0}, y{0}, z{0}
        , r{1}, g{1}, b{1}, a{1}
        , size{1}, age{0}, max_age{1}
    {}

    Particle(float px, float py, float pz, float life) noexcept
        : x{px}, y{py}, z{pz}
        , r{1}, g{1}, b{1}, a{1}
        , size{1}, age{0}, max_age{life}
    {}

    ~Particle() noexcept = default;
};

static_assert(sizeof(Particle) == 10 * sizeof(float),
    "Particle layout assumption check");
static_assert(sizeof(Particle) >= sizeof(void*),
    "Particle too small for pool");

// ---------------------------------------------------------------------------
// GameEntity
//
// Larger object. Represents a scene entity with transform, health, ID.
// ---------------------------------------------------------------------------
struct GameEntity {
    uint64_t id;
    float    position[3];
    float    rotation[4];   // quaternion
    float    scale[3];
    float    health;
    float    max_health;
    uint32_t type;
    uint32_t flags;
    char     name[32];

    GameEntity() noexcept
        : id{0}
        , position{0,0,0}
        , rotation{0,0,0,1}
        , scale{1,1,1}
        , health{100}
        , max_health{100}
        , type{0}
        , flags{0}
        , name{}
    {}

    explicit GameEntity(uint64_t eid, const char* n) noexcept
        : id{eid}
        , position{0,0,0}
        , rotation{0,0,0,1}
        , scale{1,1,1}
        , health{100}
        , max_health{100}
        , type{0}
        , flags{0}
        , name{}
    {
        // Safe truncating copy — name is fixed size
        std::strncpy(name, n, sizeof(name) - 1);
        name[sizeof(name) - 1] = '\0';
    }

    ~GameEntity() noexcept = default;
};

} // namespace fp::examples
