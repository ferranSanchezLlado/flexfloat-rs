window.BENCHMARK_DATA = {
  "lastUpdate": 1780913536943,
  "repoUrl": "https://github.com/ferranSanchezLlado/flexfloat-rs",
  "entries": {
    "FlexFloat (windows-latest)": [
      {
        "commit": {
          "author": {
            "email": "31955699+ferranSanchezLlado@users.noreply.github.com",
            "name": "Ferran Sanchez Llado",
            "username": "ferranSanchezLlado"
          },
          "committer": {
            "email": "31955699+ferranSanchezLlado@users.noreply.github.com",
            "name": "Ferran Sanchez Llado",
            "username": "ferranSanchezLlado"
          },
          "distinct": true,
          "id": "bed89bb85f0e933f5e505a7e4e0ca73cdee329da",
          "message": "refactor(arithmetic): variable-width fraction, generalise kernels, fix serialisation\n\n- normalize.rs: build_finite_result now grows the fraction field alongside\n  the exponent on overflow (n = 1 + e + m formula); align_mantissa extracted\n  as a named helper; extract_mantissa_and_exponent subnormal exponent\n  generalised (no longer hardcoded to -1022); unit_mantissa and\n  frac_bits_for_exp_width added as private helpers\n\n- mul_div.rs: both kernels widen operands to a common mant_len via\n  align_mantissa before computing, preserving precision for grown-format\n  values; hardcoded 52/53-bit indices replaced with mant_len-relative ones\n\n- add_sub.rs: import formatting; mut rhs removed from sub; debug_assert\n  relocated to after operand-swap; Sign-match exp_diff conversion simplified\n  to BigInt::try_into; unnecessary braced scope removed\n\n- converter.rs: to_le_bytes/to_be_bytes return (bytes, exp_bits, frac_bits);\n  from_le_bytes/from_be_bytes accept frac_bits for correct grown-format\n  round-trips; to_f64 uses min_exp() instead of hardcoded -1022;\n  from_int overflow guard fixed\n\n- internal.rs: exponent_bits_for_total moved here as pub(crate);\n  fraction_bits_for_total removed (unused)\n\n- docs: README byte serialisation example updated; CHANGELOG [Unreleased]\n  added; normalize.rs, mul_div.rs, arithmetic/mod.rs, add_sub.rs module\n  docs updated; lib.rs and README tables aligned",
          "timestamp": "2026-06-08T11:14:09+02:00",
          "tree_id": "5abd6b7f58802d5f56d1d81830ef812b1e43cf61",
          "url": "https://github.com/ferranSanchezLlado/flexfloat-rs/commit/bed89bb85f0e933f5e505a7e4e0ca73cdee329da"
        },
        "date": 1780911406309,
        "tool": "cargo",
        "benches": [
          {
            "name": "agm_pi/boolbitarray",
            "value": 3222139,
            "range": "± 60128",
            "unit": "ns/iter"
          },
          {
            "name": "agm_pi/f64_baseline",
            "value": 38,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "agm_pi/usizebitarray",
            "value": 1495385,
            "range": "± 10177",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/append_repeated/1024",
            "value": 35315,
            "range": "± 360",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/append_repeated/256",
            "value": 29474,
            "range": "± 288",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/append_repeated/64",
            "value": 28853,
            "range": "± 285",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bits/1024",
            "value": 13966,
            "range": "± 187",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bits/256",
            "value": 12997,
            "range": "± 1268",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bits/64",
            "value": 11075,
            "range": "± 178",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bytes/1024",
            "value": 374693,
            "range": "± 2977",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bytes/256",
            "value": 111323,
            "range": "± 13475",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bytes/64",
            "value": 33425,
            "range": "± 966",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/get_range/1024",
            "value": 15396,
            "range": "± 206",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/get_range/256",
            "value": 14469,
            "range": "± 1289",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/get_range/64",
            "value": 12537,
            "range": "± 520",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/shift_fixed/1024",
            "value": 20030,
            "range": "± 514",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/shift_fixed/256",
            "value": 13753,
            "range": "± 584",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/shift_fixed/64",
            "value": 12628,
            "range": "± 75",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/to_bytes/1024",
            "value": 179088,
            "range": "± 2850",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/to_bytes/256",
            "value": 54792,
            "range": "± 2145",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/to_bytes/64",
            "value": 21361,
            "range": "± 1267",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/append_repeated/1024",
            "value": 31221,
            "range": "± 1850",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/append_repeated/256",
            "value": 30319,
            "range": "± 1716",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/append_repeated/64",
            "value": 30507,
            "range": "± 2957",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bits/1024",
            "value": 247825,
            "range": "± 10644",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bits/256",
            "value": 120023,
            "range": "± 15672",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bits/64",
            "value": 35461,
            "range": "± 2214",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bytes/1024",
            "value": 12217,
            "range": "± 47",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bytes/256",
            "value": 11718,
            "range": "± 680",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bytes/64",
            "value": 11488,
            "range": "± 451",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/get_range/1024",
            "value": 14645,
            "range": "± 551",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/get_range/256",
            "value": 13683,
            "range": "± 1161",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/get_range/64",
            "value": 13816,
            "range": "± 476",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/shift_fixed/1024",
            "value": 559508,
            "range": "± 16365",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/shift_fixed/256",
            "value": 139031,
            "range": "± 764",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/shift_fixed/64",
            "value": 36795,
            "range": "± 2518",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/to_bytes/1024",
            "value": 12684,
            "range": "± 665",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/to_bytes/256",
            "value": 12117,
            "range": "± 721",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/to_bytes/64",
            "value": 11416,
            "range": "± 759",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_bool/add",
            "value": 7237910,
            "range": "± 439337",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_bool/from_f64",
            "value": 939205,
            "range": "± 60766",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_bool/mul",
            "value": 8754953,
            "range": "± 845338",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_bool/sqrt",
            "value": 683040903,
            "range": "± 858582469",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_bool/to_f64",
            "value": 988405,
            "range": "± 108535",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_f64_baseline/add",
            "value": 5180,
            "range": "± 26",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_f64_baseline/from_f64",
            "value": 2656,
            "range": "± 189",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_f64_baseline/mul",
            "value": 5184,
            "range": "± 45",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_f64_baseline/sqrt",
            "value": 11526,
            "range": "± 22",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_f64_baseline/to_f64",
            "value": 2754,
            "range": "± 125",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_usize/add",
            "value": 5330369,
            "range": "± 186567",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_usize/from_f64",
            "value": 607833,
            "range": "± 26256",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_usize/mul",
            "value": 6050031,
            "range": "± 349060",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_usize/sqrt",
            "value": 372589151,
            "range": "± 10113178",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_usize/to_f64",
            "value": 2227595,
            "range": "± 129907",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "31955699+ferranSanchezLlado@users.noreply.github.com",
            "name": "Ferran Sanchez Llado",
            "username": "ferranSanchezLlado"
          },
          "committer": {
            "email": "31955699+ferranSanchezLlado@users.noreply.github.com",
            "name": "Ferran Sanchez Llado",
            "username": "ferranSanchezLlado"
          },
          "distinct": true,
          "id": "aa0236a59280a69bdd8b96dd53f7a672a5719a17",
          "message": "ci: pull --rebase before modifying files to avoid unstaged-changes error",
          "timestamp": "2026-06-08T11:42:47+02:00",
          "tree_id": "bdbbb90bbdfccbb3798467f5594b23e4f72e78b7",
          "url": "https://github.com/ferranSanchezLlado/flexfloat-rs/commit/aa0236a59280a69bdd8b96dd53f7a672a5719a17"
        },
        "date": 1780913175013,
        "tool": "cargo",
        "benches": [
          {
            "name": "agm_pi/boolbitarray",
            "value": 2603043,
            "range": "± 48908",
            "unit": "ns/iter"
          },
          {
            "name": "agm_pi/f64_baseline",
            "value": 33,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "agm_pi/usizebitarray",
            "value": 1183199,
            "range": "± 13713",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/append_repeated/1024",
            "value": 32735,
            "range": "± 3634",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/append_repeated/256",
            "value": 25071,
            "range": "± 474",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/append_repeated/64",
            "value": 22732,
            "range": "± 408",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bits/1024",
            "value": 11995,
            "range": "± 216",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bits/256",
            "value": 10407,
            "range": "± 1337",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bits/64",
            "value": 8607,
            "range": "± 124",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bytes/1024",
            "value": 221335,
            "range": "± 7641",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bytes/256",
            "value": 62940,
            "range": "± 2204",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bytes/64",
            "value": 21410,
            "range": "± 403",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/get_range/1024",
            "value": 13517,
            "range": "± 246",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/get_range/256",
            "value": 11601,
            "range": "± 134",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/get_range/64",
            "value": 11395,
            "range": "± 228",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/shift_fixed/1024",
            "value": 16864,
            "range": "± 642",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/shift_fixed/256",
            "value": 10651,
            "range": "± 555",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/shift_fixed/64",
            "value": 9773,
            "range": "± 232",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/to_bytes/1024",
            "value": 150034,
            "range": "± 4076",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/to_bytes/256",
            "value": 42799,
            "range": "± 1055",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/to_bytes/64",
            "value": 16921,
            "range": "± 1189",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/append_repeated/1024",
            "value": 24033,
            "range": "± 533",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/append_repeated/256",
            "value": 22651,
            "range": "± 199",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/append_repeated/64",
            "value": 24331,
            "range": "± 1757",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bits/1024",
            "value": 193420,
            "range": "± 8494",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bits/256",
            "value": 58233,
            "range": "± 1562",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bits/64",
            "value": 27183,
            "range": "± 592",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bytes/1024",
            "value": 9989,
            "range": "± 128",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bytes/256",
            "value": 9323,
            "range": "± 140",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bytes/64",
            "value": 9345,
            "range": "± 199",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/get_range/1024",
            "value": 11954,
            "range": "± 126",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/get_range/256",
            "value": 11692,
            "range": "± 867",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/get_range/64",
            "value": 11694,
            "range": "± 338",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/shift_fixed/1024",
            "value": 467587,
            "range": "± 5501",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/shift_fixed/256",
            "value": 117193,
            "range": "± 4158",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/shift_fixed/64",
            "value": 29453,
            "range": "± 1273",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/to_bytes/1024",
            "value": 9596,
            "range": "± 398",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/to_bytes/256",
            "value": 9191,
            "range": "± 118",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/to_bytes/64",
            "value": 9239,
            "range": "± 781",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_bool/add",
            "value": 4971676,
            "range": "± 562327",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_bool/from_f64",
            "value": 645328,
            "range": "± 14776",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_bool/mul",
            "value": 6868839,
            "range": "± 161923",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_bool/sqrt",
            "value": 467075062,
            "range": "± 16382238",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_bool/to_f64",
            "value": 659972,
            "range": "± 12349",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_f64_baseline/add",
            "value": 4659,
            "range": "± 95",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_f64_baseline/from_f64",
            "value": 2329,
            "range": "± 71",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_f64_baseline/mul",
            "value": 4700,
            "range": "± 137",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_f64_baseline/sqrt",
            "value": 10343,
            "range": "± 193",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_f64_baseline/to_f64",
            "value": 2300,
            "range": "± 68",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_usize/add",
            "value": 4321168,
            "range": "± 102185",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_usize/from_f64",
            "value": 467577,
            "range": "± 15939",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_usize/mul",
            "value": 4933651,
            "range": "± 55799",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_usize/sqrt",
            "value": 287207770,
            "range": "± 5664216",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_usize/to_f64",
            "value": 1682854,
            "range": "± 31935",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "31955699+ferranSanchezLlado@users.noreply.github.com",
            "name": "Ferran Sanchez Llado",
            "username": "ferranSanchezLlado"
          },
          "committer": {
            "email": "31955699+ferranSanchezLlado@users.noreply.github.com",
            "name": "Ferran Sanchez Llado",
            "username": "ferranSanchezLlado"
          },
          "distinct": true,
          "id": "df2f8389d7f405ad500c5fd77ce566b5725018c8",
          "message": "ci: pull --rebase before modifying files to avoid unstaged-changes error",
          "timestamp": "2026-06-08T11:49:59+02:00",
          "tree_id": "df6a3c15cd39aeaa0a3723329a12a7e7c8ab94cd",
          "url": "https://github.com/ferranSanchezLlado/flexfloat-rs/commit/df2f8389d7f405ad500c5fd77ce566b5725018c8"
        },
        "date": 1780913535096,
        "tool": "cargo",
        "benches": [
          {
            "name": "agm_pi/boolbitarray",
            "value": 3252024,
            "range": "± 53541",
            "unit": "ns/iter"
          },
          {
            "name": "agm_pi/f64_baseline",
            "value": 43,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "agm_pi/usizebitarray",
            "value": 1515317,
            "range": "± 9074",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/append_repeated/1024",
            "value": 38872,
            "range": "± 2526",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/append_repeated/256",
            "value": 33576,
            "range": "± 3064",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/append_repeated/64",
            "value": 29500,
            "range": "± 651",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bits/1024",
            "value": 14720,
            "range": "± 138",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bits/256",
            "value": 11511,
            "range": "± 55",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bits/64",
            "value": 11165,
            "range": "± 256",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bytes/1024",
            "value": 278393,
            "range": "± 9289",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bytes/256",
            "value": 81135,
            "range": "± 4491",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bytes/64",
            "value": 26884,
            "range": "± 268",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/get_range/1024",
            "value": 16473,
            "range": "± 179",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/get_range/256",
            "value": 15025,
            "range": "± 1503",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/get_range/64",
            "value": 14284,
            "range": "± 197",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/shift_fixed/1024",
            "value": 20912,
            "range": "± 1353",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/shift_fixed/256",
            "value": 15374,
            "range": "± 1969",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/shift_fixed/64",
            "value": 12775,
            "range": "± 92",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/to_bytes/1024",
            "value": 196144,
            "range": "± 20108",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/to_bytes/256",
            "value": 54535,
            "range": "± 1494",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/to_bytes/64",
            "value": 21557,
            "range": "± 475",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/append_repeated/1024",
            "value": 31182,
            "range": "± 635",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/append_repeated/256",
            "value": 30625,
            "range": "± 374",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/append_repeated/64",
            "value": 30372,
            "range": "± 706",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bits/1024",
            "value": 264744,
            "range": "± 28024",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bits/256",
            "value": 76165,
            "range": "± 4020",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bits/64",
            "value": 34184,
            "range": "± 447",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bytes/1024",
            "value": 12692,
            "range": "± 67",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bytes/256",
            "value": 11965,
            "range": "± 111",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bytes/64",
            "value": 11880,
            "range": "± 88",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/get_range/1024",
            "value": 15868,
            "range": "± 1009",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/get_range/256",
            "value": 15752,
            "range": "± 1670",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/get_range/64",
            "value": 15721,
            "range": "± 423",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/shift_fixed/1024",
            "value": 603903,
            "range": "± 8996",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/shift_fixed/256",
            "value": 151874,
            "range": "± 6127",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/shift_fixed/64",
            "value": 37764,
            "range": "± 387",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/to_bytes/1024",
            "value": 12769,
            "range": "± 729",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/to_bytes/256",
            "value": 11890,
            "range": "± 646",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/to_bytes/64",
            "value": 11893,
            "range": "± 340",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_bool/add",
            "value": 7327422,
            "range": "± 507236",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_bool/from_f64",
            "value": 818996,
            "range": "± 2232",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_bool/mul",
            "value": 10814276,
            "range": "± 896943",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_bool/sqrt",
            "value": 696632780,
            "range": "± 895595758",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_bool/to_f64",
            "value": 878960,
            "range": "± 41756",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_f64_baseline/add",
            "value": 5770,
            "range": "± 16",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_f64_baseline/from_f64",
            "value": 2968,
            "range": "± 185",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_f64_baseline/mul",
            "value": 5773,
            "range": "± 29",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_f64_baseline/sqrt",
            "value": 12991,
            "range": "± 33",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_f64_baseline/to_f64",
            "value": 2944,
            "range": "± 98",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_usize/add",
            "value": 5578210,
            "range": "± 229211",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_usize/from_f64",
            "value": 608208,
            "range": "± 37712",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_usize/mul",
            "value": 6396455,
            "range": "± 240976",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_usize/sqrt",
            "value": 376482358,
            "range": "± 7356961",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_usize/to_f64",
            "value": 2179984,
            "range": "± 172104",
            "unit": "ns/iter"
          }
        ]
      }
    ]
  }
}