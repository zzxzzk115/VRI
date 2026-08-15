// Backend-neutral tests for the shared conversion skeleton (core/conversion_map.h).
//
// These lock in the exact semantics every backend's conversion table now relies
// on: MapOr must behave identically to a `switch (x) { case K: return V; default:
// return F; }`, and MapFlags identically to an `if (x & K) r |= V;` chain. The
// tables use local mock enums, so no backend / native SDK is required - this runs
// on every configuration.
#include <doctest/doctest.h>

#include "core/conversion_map.h"

namespace
{
    enum MockEnum
    {
        Mock_A = 0,
        Mock_B = 1,
        Mock_C = 2,
        Mock_Unmapped = 99,
    };

    enum MockFlagBits
    {
        Flag_None = 0,
        Flag_X    = 1 << 0,
        Flag_Y    = 1 << 1,
        Flag_Z    = 1 << 2,
        Flag_W    = 1 << 3,
    };

    // The hand-written shapes MapOr / MapFlags are meant to replace, kept here as
    // the oracle the table-driven version is checked against.
    int SwitchOracle(MockEnum e)
    {
        switch (e)
        {
            case Mock_A: return 10;
            case Mock_B: return 20;
            case Mock_C: return 30;
            default:     return -1;
        }
    }

    unsigned IfChainOracle(unsigned f)
    {
        unsigned r = 0;
        if (f & Flag_X) r |= 0x100;
        if (f & Flag_Y) r |= 0x200;
        if (f & Flag_Z) r |= 0x400 | 0x800; // one VRI bit -> several native bits
        return r;
    }
} // namespace

TEST_CASE("MapOr maps known keys and falls back on unmapped")
{
    static constexpr vri::ConvRow<MockEnum, int> kTable[] = {
        {Mock_A, 10},
        {Mock_B, 20},
        {Mock_C, 30},
    };

    CHECK(vri::MapOr(Mock_A, kTable, -1) == 10);
    CHECK(vri::MapOr(Mock_B, kTable, -1) == 20);
    CHECK(vri::MapOr(Mock_C, kTable, -1) == 30);
    CHECK(vri::MapOr(Mock_Unmapped, kTable, -1) == -1);

    // Exhaustively equivalent to the switch oracle it replaces.
    for (int raw = -5; raw < 120; ++raw)
    {
        const auto e = static_cast<MockEnum>(raw);
        CHECK(vri::MapOr(e, kTable, -1) == SwitchOracle(e));
    }
}

TEST_CASE("MapOr carries aggregate (struct) values")
{
    struct Info
    {
        int   a;
        float b;
    };
    static constexpr vri::ConvRow<MockEnum, Info> kTable[] = {
        {Mock_A, {1, 1.5f}},
        {Mock_B, {2, 2.5f}},
    };

    CHECK(vri::MapOr(Mock_B, kTable, Info {0, 0.0f}).a == 2);
    CHECK(vri::MapOr(Mock_B, kTable, Info {0, 0.0f}).b == doctest::Approx(2.5f));
    CHECK(vri::MapOr(Mock_Unmapped, kTable, Info {7, 7.0f}).a == 7); // fallback returned by value
}

TEST_CASE("MapFlags ORs mapped bits and is order/duplicate independent")
{
    static constexpr vri::ConvRow<unsigned, unsigned> kTable[] = {
        {Flag_X, 0x100},
        {Flag_Y, 0x200},
        {Flag_Z, 0x400 | 0x800},
    };

    CHECK(vri::MapFlags(0u, kTable) == 0u);
    CHECK(vri::MapFlags((unsigned)Flag_X, kTable) == 0x100u);
    CHECK(vri::MapFlags((unsigned)(Flag_X | Flag_Z), kTable) == (0x100u | 0x400u | 0x800u));
    CHECK(vri::MapFlags((unsigned)Flag_W, kTable) == 0u); // bit with no row contributes nothing

    // Exhaustively equivalent to the if-chain oracle it replaces.
    for (unsigned f = 0; f < 16; ++f)
        CHECK(vri::MapFlags(f, kTable) == IfChainOracle(f));
}

TEST_CASE("MapFlags honors a non-zero 'zero' seed")
{
    static constexpr vri::ConvRow<unsigned, unsigned> kTable[] = {
        {Flag_X, 0x1},
    };
    CHECK(vri::MapFlags(0u, kTable, 0xF000u) == 0xF000u);
    CHECK(vri::MapFlags((unsigned)Flag_X, kTable, 0xF000u) == 0xF001u);
}
