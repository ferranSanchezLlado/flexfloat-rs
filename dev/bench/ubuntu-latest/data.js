window.BENCHMARK_DATA = {
  "lastUpdate": 1780913401479,
  "repoUrl": "https://github.com/ferranSanchezLlado/flexfloat-rs",
  "entries": {
    "FlexFloat (ubuntu-latest)": [
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
          "id": "24b7c24cdcba8a1a3ba6e4ad347481d06b6266ee",
          "message": "ci: create gh-pages branch automatically if it does not exist",
          "timestamp": "2026-06-05T11:56:51+02:00",
          "tree_id": "53cf1a40ae38fe0975f8c27419b34edecfef3ef5",
          "url": "https://github.com/ferranSanchezLlado/flexfloat-rs/commit/24b7c24cdcba8a1a3ba6e4ad347481d06b6266ee"
        },
        "date": 1780654596155,
        "tool": "cargo",
        "benches": [
          {
            "name": "agm_pi/BoolBitArray",
            "value": 1742683,
            "range": "± 21617",
            "unit": "ns/iter"
          },
          {
            "name": "agm_pi/UsizeBitArray",
            "value": 904812,
            "range": "± 13843",
            "unit": "ns/iter"
          },
          {
            "name": "agm_pi/f64_baseline",
            "value": 42,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/append_repeated/1024",
            "value": 24301,
            "range": "± 762",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/append_repeated/256",
            "value": 20291,
            "range": "± 1060",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/append_repeated/64",
            "value": 14476,
            "range": "± 288",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bits/1024",
            "value": 6320,
            "range": "± 66",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bits/256",
            "value": 4847,
            "range": "± 71",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bits/64",
            "value": 4608,
            "range": "± 50",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bytes/1024",
            "value": 299326,
            "range": "± 2392",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bytes/256",
            "value": 75205,
            "range": "± 3002",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bytes/64",
            "value": 19912,
            "range": "± 493",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/get_range/1024",
            "value": 7791,
            "range": "± 55",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/get_range/256",
            "value": 6680,
            "range": "± 32",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/get_range/64",
            "value": 6319,
            "range": "± 20",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/shift_fixed/1024",
            "value": 10900,
            "range": "± 151",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/shift_fixed/256",
            "value": 6198,
            "range": "± 71",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/shift_fixed/64",
            "value": 5962,
            "range": "± 53",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/to_bytes/1024",
            "value": 186240,
            "range": "± 5252",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/to_bytes/256",
            "value": 49083,
            "range": "± 466",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/to_bytes/64",
            "value": 16817,
            "range": "± 545",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/append_repeated/1024",
            "value": 19766,
            "range": "± 338",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/append_repeated/256",
            "value": 13914,
            "range": "± 213",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/append_repeated/64",
            "value": 13757,
            "range": "± 75",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bits/1024",
            "value": 268355,
            "range": "± 3347",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bits/256",
            "value": 68838,
            "range": "± 3193",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bits/64",
            "value": 26625,
            "range": "± 126",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bytes/1024",
            "value": 11761,
            "range": "± 260",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bytes/256",
            "value": 7318,
            "range": "± 66",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bytes/64",
            "value": 7408,
            "range": "± 105",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/get_range/1024",
            "value": 8345,
            "range": "± 80",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/get_range/256",
            "value": 8393,
            "range": "± 17",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/get_range/64",
            "value": 9535,
            "range": "± 90",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/shift_fixed/1024",
            "value": 590789,
            "range": "± 9254",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/shift_fixed/256",
            "value": 141770,
            "range": "± 929",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/shift_fixed/64",
            "value": 29704,
            "range": "± 248",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/to_bytes/1024",
            "value": 12203,
            "range": "± 183",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/to_bytes/256",
            "value": 7051,
            "range": "± 65",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/to_bytes/64",
            "value": 6866,
            "range": "± 63",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_bool/add",
            "value": 3246402,
            "range": "± 6658",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_bool/from_f64",
            "value": 470451,
            "range": "± 16832",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_bool/mul",
            "value": 5548586,
            "range": "± 140321",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_bool/sqrt",
            "value": 422030068,
            "range": "± 1499764",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_bool/to_f64",
            "value": 466698,
            "range": "± 6642",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_f64_baseline/add",
            "value": 5756,
            "range": "± 69",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_f64_baseline/from_f64",
            "value": 2883,
            "range": "± 13",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_f64_baseline/mul",
            "value": 5745,
            "range": "± 50",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_f64_baseline/sqrt",
            "value": 12949,
            "range": "± 8",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_f64_baseline/to_f64",
            "value": 2898,
            "range": "± 26",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_usize/add",
            "value": 3295756,
            "range": "± 16869",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_usize/from_f64",
            "value": 333036,
            "range": "± 1594",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_usize/mul",
            "value": 3588584,
            "range": "± 9954",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_usize/sqrt",
            "value": 220709637,
            "range": "± 1322506",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_usize/to_f64",
            "value": 1068256,
            "range": "± 8162",
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
          "id": "dd6e19696df0688d49c53ea0df82ca0547899636",
          "message": "ci: fix gh-pages init checkout on non-ubuntu runners",
          "timestamp": "2026-06-05T13:09:07+02:00",
          "tree_id": "5f408cf0bd9dbd285371e621843f8eded50e7336",
          "url": "https://github.com/ferranSanchezLlado/flexfloat-rs/commit/dd6e19696df0688d49c53ea0df82ca0547899636"
        },
        "date": 1780659088485,
        "tool": "cargo",
        "benches": [
          {
            "name": "agm_pi/BoolBitArray",
            "value": 1711269,
            "range": "± 4724",
            "unit": "ns/iter"
          },
          {
            "name": "agm_pi/UsizeBitArray",
            "value": 927680,
            "range": "± 4262",
            "unit": "ns/iter"
          },
          {
            "name": "agm_pi/f64_baseline",
            "value": 37,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/append_repeated/1024",
            "value": 21357,
            "range": "± 138",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/append_repeated/256",
            "value": 14656,
            "range": "± 72",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/append_repeated/64",
            "value": 16141,
            "range": "± 159",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bits/1024",
            "value": 6379,
            "range": "± 96",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bits/256",
            "value": 4619,
            "range": "± 201",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bits/64",
            "value": 4677,
            "range": "± 204",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bytes/1024",
            "value": 362545,
            "range": "± 2469",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bytes/256",
            "value": 93136,
            "range": "± 1072",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bytes/64",
            "value": 26448,
            "range": "± 1034",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/get_range/1024",
            "value": 7165,
            "range": "± 41",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/get_range/256",
            "value": 6001,
            "range": "± 42",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/get_range/64",
            "value": 5431,
            "range": "± 14",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/shift_fixed/1024",
            "value": 11124,
            "range": "± 97",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/shift_fixed/256",
            "value": 6329,
            "range": "± 83",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/shift_fixed/64",
            "value": 5531,
            "range": "± 39",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/to_bytes/1024",
            "value": 175923,
            "range": "± 3070",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/to_bytes/256",
            "value": 49845,
            "range": "± 662",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/to_bytes/64",
            "value": 16358,
            "range": "± 217",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/append_repeated/1024",
            "value": 18485,
            "range": "± 229",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/append_repeated/256",
            "value": 13312,
            "range": "± 75",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/append_repeated/64",
            "value": 13445,
            "range": "± 70",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bits/1024",
            "value": 244603,
            "range": "± 1806",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bits/256",
            "value": 68254,
            "range": "± 585",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bits/64",
            "value": 26416,
            "range": "± 371",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bytes/1024",
            "value": 13599,
            "range": "± 108",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bytes/256",
            "value": 7200,
            "range": "± 90",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bytes/64",
            "value": 7236,
            "range": "± 272",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/get_range/1024",
            "value": 6374,
            "range": "± 23",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/get_range/256",
            "value": 6253,
            "range": "± 101",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/get_range/64",
            "value": 8232,
            "range": "± 431",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/shift_fixed/1024",
            "value": 550855,
            "range": "± 1523",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/shift_fixed/256",
            "value": 133272,
            "range": "± 1725",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/shift_fixed/64",
            "value": 29745,
            "range": "± 426",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/to_bytes/1024",
            "value": 13287,
            "range": "± 141",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/to_bytes/256",
            "value": 7033,
            "range": "± 22",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/to_bytes/64",
            "value": 7147,
            "range": "± 303",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_bool/add",
            "value": 3301883,
            "range": "± 12369",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_bool/from_f64",
            "value": 592260,
            "range": "± 5003",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_bool/mul",
            "value": 5175692,
            "range": "± 21122",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_bool/sqrt",
            "value": 410036871,
            "range": "± 4266985",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_bool/to_f64",
            "value": 502125,
            "range": "± 5363",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_f64_baseline/add",
            "value": 5188,
            "range": "± 23",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_f64_baseline/from_f64",
            "value": 2558,
            "range": "± 34",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_f64_baseline/mul",
            "value": 5198,
            "range": "± 49",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_f64_baseline/sqrt",
            "value": 11490,
            "range": "± 15",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_f64_baseline/to_f64",
            "value": 2718,
            "range": "± 15",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_usize/add",
            "value": 3243116,
            "range": "± 25843",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_usize/from_f64",
            "value": 341926,
            "range": "± 2803",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_usize/mul",
            "value": 3583113,
            "range": "± 22319",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_usize/sqrt",
            "value": 220243764,
            "range": "± 6394578",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_usize/to_f64",
            "value": 1214871,
            "range": "± 12475",
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
          "id": "cd57b43976a38b9459894e4eb10a844cbe1e5554",
          "message": "ci: pull --rebase before release commit to avoid push rejection",
          "timestamp": "2026-06-05T16:01:15+02:00",
          "tree_id": "630fc8761716457a22d14da234e2e02c38426920",
          "url": "https://github.com/ferranSanchezLlado/flexfloat-rs/commit/cd57b43976a38b9459894e4eb10a844cbe1e5554"
        },
        "date": 1780669346362,
        "tool": "cargo",
        "benches": [
          {
            "name": "agm_pi/BoolBitArray",
            "value": 1771583,
            "range": "± 12449",
            "unit": "ns/iter"
          },
          {
            "name": "agm_pi/UsizeBitArray",
            "value": 922694,
            "range": "± 7498",
            "unit": "ns/iter"
          },
          {
            "name": "agm_pi/f64_baseline",
            "value": 42,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/append_repeated/1024",
            "value": 23013,
            "range": "± 295",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/append_repeated/256",
            "value": 19018,
            "range": "± 257",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/append_repeated/64",
            "value": 17791,
            "range": "± 169",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bits/1024",
            "value": 6197,
            "range": "± 53",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bits/256",
            "value": 4615,
            "range": "± 77",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bits/64",
            "value": 4607,
            "range": "± 32",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bytes/1024",
            "value": 282883,
            "range": "± 3265",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bytes/256",
            "value": 74720,
            "range": "± 1514",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bytes/64",
            "value": 20359,
            "range": "± 708",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/get_range/1024",
            "value": 7939,
            "range": "± 35",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/get_range/256",
            "value": 6714,
            "range": "± 103",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/get_range/64",
            "value": 6317,
            "range": "± 48",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/shift_fixed/1024",
            "value": 10235,
            "range": "± 162",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/shift_fixed/256",
            "value": 6158,
            "range": "± 129",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/shift_fixed/64",
            "value": 6156,
            "range": "± 26",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/to_bytes/1024",
            "value": 185230,
            "range": "± 2455",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/to_bytes/256",
            "value": 52372,
            "range": "± 700",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/to_bytes/64",
            "value": 17592,
            "range": "± 409",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/append_repeated/1024",
            "value": 19162,
            "range": "± 130",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/append_repeated/256",
            "value": 13796,
            "range": "± 45",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/append_repeated/64",
            "value": 13701,
            "range": "± 156",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bits/1024",
            "value": 263287,
            "range": "± 8185",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bits/256",
            "value": 73070,
            "range": "± 2917",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bits/64",
            "value": 26570,
            "range": "± 306",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bytes/1024",
            "value": 11466,
            "range": "± 335",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bytes/256",
            "value": 7336,
            "range": "± 159",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bytes/64",
            "value": 7228,
            "range": "± 73",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/get_range/1024",
            "value": 8242,
            "range": "± 46",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/get_range/256",
            "value": 8386,
            "range": "± 30",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/get_range/64",
            "value": 9418,
            "range": "± 48",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/shift_fixed/1024",
            "value": 590040,
            "range": "± 1626",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/shift_fixed/256",
            "value": 141345,
            "range": "± 1238",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/shift_fixed/64",
            "value": 29707,
            "range": "± 238",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/to_bytes/1024",
            "value": 12053,
            "range": "± 470",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/to_bytes/256",
            "value": 7049,
            "range": "± 71",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/to_bytes/64",
            "value": 6873,
            "range": "± 81",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_bool/add",
            "value": 3280435,
            "range": "± 12127",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_bool/from_f64",
            "value": 455273,
            "range": "± 13781",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_bool/mul",
            "value": 5567654,
            "range": "± 10344",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_bool/sqrt",
            "value": 427717234,
            "range": "± 3146509",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_bool/to_f64",
            "value": 466572,
            "range": "± 7011",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_f64_baseline/add",
            "value": 5743,
            "range": "± 39",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_f64_baseline/from_f64",
            "value": 2882,
            "range": "± 23",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_f64_baseline/mul",
            "value": 5748,
            "range": "± 63",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_f64_baseline/sqrt",
            "value": 12981,
            "range": "± 113",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_f64_baseline/to_f64",
            "value": 2900,
            "range": "± 37",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_usize/add",
            "value": 3412400,
            "range": "± 12744",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_usize/from_f64",
            "value": 333062,
            "range": "± 5117",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_usize/mul",
            "value": 3697027,
            "range": "± 15153",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_usize/sqrt",
            "value": 228562523,
            "range": "± 1353497",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_usize/to_f64",
            "value": 1057453,
            "range": "± 14803",
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
          "id": "bed89bb85f0e933f5e505a7e4e0ca73cdee329da",
          "message": "refactor(arithmetic): variable-width fraction, generalise kernels, fix serialisation\n\n- normalize.rs: build_finite_result now grows the fraction field alongside\n  the exponent on overflow (n = 1 + e + m formula); align_mantissa extracted\n  as a named helper; extract_mantissa_and_exponent subnormal exponent\n  generalised (no longer hardcoded to -1022); unit_mantissa and\n  frac_bits_for_exp_width added as private helpers\n\n- mul_div.rs: both kernels widen operands to a common mant_len via\n  align_mantissa before computing, preserving precision for grown-format\n  values; hardcoded 52/53-bit indices replaced with mant_len-relative ones\n\n- add_sub.rs: import formatting; mut rhs removed from sub; debug_assert\n  relocated to after operand-swap; Sign-match exp_diff conversion simplified\n  to BigInt::try_into; unnecessary braced scope removed\n\n- converter.rs: to_le_bytes/to_be_bytes return (bytes, exp_bits, frac_bits);\n  from_le_bytes/from_be_bytes accept frac_bits for correct grown-format\n  round-trips; to_f64 uses min_exp() instead of hardcoded -1022;\n  from_int overflow guard fixed\n\n- internal.rs: exponent_bits_for_total moved here as pub(crate);\n  fraction_bits_for_total removed (unused)\n\n- docs: README byte serialisation example updated; CHANGELOG [Unreleased]\n  added; normalize.rs, mul_div.rs, arithmetic/mod.rs, add_sub.rs module\n  docs updated; lib.rs and README tables aligned",
          "timestamp": "2026-06-08T11:14:09+02:00",
          "tree_id": "5abd6b7f58802d5f56d1d81830ef812b1e43cf61",
          "url": "https://github.com/ferranSanchezLlado/flexfloat-rs/commit/bed89bb85f0e933f5e505a7e4e0ca73cdee329da"
        },
        "date": 1780911244410,
        "tool": "cargo",
        "benches": [
          {
            "name": "agm_pi/BoolBitArray",
            "value": 1736943,
            "range": "± 13073",
            "unit": "ns/iter"
          },
          {
            "name": "agm_pi/UsizeBitArray",
            "value": 936226,
            "range": "± 9351",
            "unit": "ns/iter"
          },
          {
            "name": "agm_pi/f64_baseline",
            "value": 37,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/append_repeated/1024",
            "value": 22454,
            "range": "± 278",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/append_repeated/256",
            "value": 19457,
            "range": "± 498",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/append_repeated/64",
            "value": 17765,
            "range": "± 432",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bits/1024",
            "value": 6661,
            "range": "± 72",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bits/256",
            "value": 4592,
            "range": "± 85",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bits/64",
            "value": 4439,
            "range": "± 45",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bytes/1024",
            "value": 353924,
            "range": "± 2937",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bytes/256",
            "value": 90727,
            "range": "± 1144",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bytes/64",
            "value": 26139,
            "range": "± 514",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/get_range/1024",
            "value": 7084,
            "range": "± 39",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/get_range/256",
            "value": 5751,
            "range": "± 66",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/get_range/64",
            "value": 5449,
            "range": "± 43",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/shift_fixed/1024",
            "value": 11589,
            "range": "± 94",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/shift_fixed/256",
            "value": 11948,
            "range": "± 32",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/shift_fixed/64",
            "value": 5536,
            "range": "± 191",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/to_bytes/1024",
            "value": 183623,
            "range": "± 3749",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/to_bytes/256",
            "value": 50816,
            "range": "± 972",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/to_bytes/64",
            "value": 16970,
            "range": "± 287",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/append_repeated/1024",
            "value": 19332,
            "range": "± 405",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/append_repeated/256",
            "value": 14238,
            "range": "± 318",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/append_repeated/64",
            "value": 14185,
            "range": "± 408",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bits/1024",
            "value": 203703,
            "range": "± 1340",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bits/256",
            "value": 60884,
            "range": "± 667",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bits/64",
            "value": 24905,
            "range": "± 553",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bytes/1024",
            "value": 12321,
            "range": "± 186",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bytes/256",
            "value": 7704,
            "range": "± 296",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bytes/64",
            "value": 7624,
            "range": "± 211",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/get_range/1024",
            "value": 6409,
            "range": "± 66",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/get_range/256",
            "value": 6352,
            "range": "± 67",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/get_range/64",
            "value": 8357,
            "range": "± 160",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/shift_fixed/1024",
            "value": 539813,
            "range": "± 1530",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/shift_fixed/256",
            "value": 132805,
            "range": "± 1029",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/shift_fixed/64",
            "value": 29545,
            "range": "± 437",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/to_bytes/1024",
            "value": 12227,
            "range": "± 205",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/to_bytes/256",
            "value": 7319,
            "range": "± 169",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/to_bytes/64",
            "value": 7421,
            "range": "± 206",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_bool/add",
            "value": 3453349,
            "range": "± 38447",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_bool/from_f64",
            "value": 595753,
            "range": "± 10909",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_bool/mul",
            "value": 5413715,
            "range": "± 63550",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_bool/sqrt",
            "value": 433369354,
            "range": "± 3173924",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_bool/to_f64",
            "value": 527132,
            "range": "± 8151",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_f64_baseline/add",
            "value": 5388,
            "range": "± 89",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_f64_baseline/from_f64",
            "value": 2592,
            "range": "± 29",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_f64_baseline/mul",
            "value": 5333,
            "range": "± 110",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_f64_baseline/sqrt",
            "value": 11737,
            "range": "± 155",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_f64_baseline/to_f64",
            "value": 2784,
            "range": "± 42",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_usize/add",
            "value": 3275726,
            "range": "± 28720",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_usize/from_f64",
            "value": 348711,
            "range": "± 8456",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_usize/mul",
            "value": 3519359,
            "range": "± 32754",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_usize/sqrt",
            "value": 226231080,
            "range": "± 2109349",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_usize/to_f64",
            "value": 1191123,
            "range": "± 15670",
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
        "date": 1780913083162,
        "tool": "cargo",
        "benches": [
          {
            "name": "agm_pi/BoolBitArray",
            "value": 1720787,
            "range": "± 9659",
            "unit": "ns/iter"
          },
          {
            "name": "agm_pi/UsizeBitArray",
            "value": 945604,
            "range": "± 9953",
            "unit": "ns/iter"
          },
          {
            "name": "agm_pi/f64_baseline",
            "value": 37,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/append_repeated/1024",
            "value": 23305,
            "range": "± 179",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/append_repeated/256",
            "value": 16725,
            "range": "± 176",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/append_repeated/64",
            "value": 17396,
            "range": "± 132",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bits/1024",
            "value": 5955,
            "range": "± 69",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bits/256",
            "value": 4601,
            "range": "± 126",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bits/64",
            "value": 4374,
            "range": "± 103",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bytes/1024",
            "value": 351848,
            "range": "± 2384",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bytes/256",
            "value": 90605,
            "range": "± 905",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bytes/64",
            "value": 25668,
            "range": "± 323",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/get_range/1024",
            "value": 7086,
            "range": "± 38",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/get_range/256",
            "value": 7331,
            "range": "± 292",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/get_range/64",
            "value": 5450,
            "range": "± 43",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/shift_fixed/1024",
            "value": 11062,
            "range": "± 112",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/shift_fixed/256",
            "value": 6100,
            "range": "± 57",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/shift_fixed/64",
            "value": 5451,
            "range": "± 57",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/to_bytes/1024",
            "value": 179927,
            "range": "± 811",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/to_bytes/256",
            "value": 50542,
            "range": "± 501",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/to_bytes/64",
            "value": 16491,
            "range": "± 240",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/append_repeated/1024",
            "value": 19289,
            "range": "± 259",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/append_repeated/256",
            "value": 13635,
            "range": "± 192",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/append_repeated/64",
            "value": 13728,
            "range": "± 193",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bits/1024",
            "value": 202517,
            "range": "± 2341",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bits/256",
            "value": 60111,
            "range": "± 339",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bits/64",
            "value": 24108,
            "range": "± 173",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bytes/1024",
            "value": 11970,
            "range": "± 214",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bytes/256",
            "value": 7297,
            "range": "± 139",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bytes/64",
            "value": 7280,
            "range": "± 89",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/get_range/1024",
            "value": 7458,
            "range": "± 45",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/get_range/256",
            "value": 6359,
            "range": "± 48",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/get_range/64",
            "value": 8299,
            "range": "± 45",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/shift_fixed/1024",
            "value": 540543,
            "range": "± 4300",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/shift_fixed/256",
            "value": 132486,
            "range": "± 1079",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/shift_fixed/64",
            "value": 29401,
            "range": "± 281",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/to_bytes/1024",
            "value": 11808,
            "range": "± 73",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/to_bytes/256",
            "value": 7002,
            "range": "± 229",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/to_bytes/64",
            "value": 7041,
            "range": "± 78",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_bool/add",
            "value": 3215592,
            "range": "± 8910",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_bool/from_f64",
            "value": 577867,
            "range": "± 6660",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_bool/mul",
            "value": 5251281,
            "range": "± 25458",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_bool/sqrt",
            "value": 418786944,
            "range": "± 1961565",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_bool/to_f64",
            "value": 499240,
            "range": "± 7389",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_f64_baseline/add",
            "value": 5191,
            "range": "± 50",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_f64_baseline/from_f64",
            "value": 2557,
            "range": "± 26",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_f64_baseline/mul",
            "value": 5195,
            "range": "± 34",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_f64_baseline/sqrt",
            "value": 11487,
            "range": "± 11",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_f64_baseline/to_f64",
            "value": 2720,
            "range": "± 15",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_usize/add",
            "value": 3226633,
            "range": "± 14724",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_usize/from_f64",
            "value": 328123,
            "range": "± 1787",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_usize/mul",
            "value": 3554496,
            "range": "± 15557",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_usize/sqrt",
            "value": 221355261,
            "range": "± 1344502",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_usize/to_f64",
            "value": 1135949,
            "range": "± 11444",
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
        "date": 1780913400964,
        "tool": "cargo",
        "benches": [
          {
            "name": "agm_pi/BoolBitArray",
            "value": 1710639,
            "range": "± 20812",
            "unit": "ns/iter"
          },
          {
            "name": "agm_pi/UsizeBitArray",
            "value": 917154,
            "range": "± 7622",
            "unit": "ns/iter"
          },
          {
            "name": "agm_pi/f64_baseline",
            "value": 37,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/append_repeated/1024",
            "value": 26042,
            "range": "± 240",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/append_repeated/256",
            "value": 17942,
            "range": "± 235",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/append_repeated/64",
            "value": 16112,
            "range": "± 170",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bits/1024",
            "value": 5743,
            "range": "± 26",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bits/256",
            "value": 4401,
            "range": "± 23",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bits/64",
            "value": 4302,
            "range": "± 55",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bytes/1024",
            "value": 352292,
            "range": "± 1925",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bytes/256",
            "value": 90926,
            "range": "± 1388",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bytes/64",
            "value": 25583,
            "range": "± 481",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/get_range/1024",
            "value": 7490,
            "range": "± 54",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/get_range/256",
            "value": 6111,
            "range": "± 34",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/get_range/64",
            "value": 5607,
            "range": "± 34",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/shift_fixed/1024",
            "value": 10773,
            "range": "± 79",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/shift_fixed/256",
            "value": 6385,
            "range": "± 180",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/shift_fixed/64",
            "value": 5371,
            "range": "± 74",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/to_bytes/1024",
            "value": 180270,
            "range": "± 2560",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/to_bytes/256",
            "value": 49760,
            "range": "± 650",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/to_bytes/64",
            "value": 16405,
            "range": "± 230",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/append_repeated/1024",
            "value": 19144,
            "range": "± 88",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/append_repeated/256",
            "value": 13447,
            "range": "± 157",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/append_repeated/64",
            "value": 13589,
            "range": "± 63",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bits/1024",
            "value": 203853,
            "range": "± 1281",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bits/256",
            "value": 60144,
            "range": "± 663",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bits/64",
            "value": 24120,
            "range": "± 241",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bytes/1024",
            "value": 12010,
            "range": "± 369",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bytes/256",
            "value": 7276,
            "range": "± 33",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bytes/64",
            "value": 7277,
            "range": "± 63",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/get_range/1024",
            "value": 6363,
            "range": "± 45",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/get_range/256",
            "value": 6299,
            "range": "± 49",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/get_range/64",
            "value": 8293,
            "range": "± 54",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/shift_fixed/1024",
            "value": 539684,
            "range": "± 1027",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/shift_fixed/256",
            "value": 132949,
            "range": "± 582",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/shift_fixed/64",
            "value": 29458,
            "range": "± 189",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/to_bytes/1024",
            "value": 11843,
            "range": "± 145",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/to_bytes/256",
            "value": 6960,
            "range": "± 31",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/to_bytes/64",
            "value": 7046,
            "range": "± 71",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_bool/add",
            "value": 3351647,
            "range": "± 15767",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_bool/from_f64",
            "value": 582037,
            "range": "± 5065",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_bool/mul",
            "value": 5294974,
            "range": "± 127640",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_bool/sqrt",
            "value": 420116771,
            "range": "± 2491469",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_bool/to_f64",
            "value": 497175,
            "range": "± 4666",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_f64_baseline/add",
            "value": 5187,
            "range": "± 27",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_f64_baseline/from_f64",
            "value": 2555,
            "range": "± 9",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_f64_baseline/mul",
            "value": 5192,
            "range": "± 47",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_f64_baseline/sqrt",
            "value": 11490,
            "range": "± 7",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_f64_baseline/to_f64",
            "value": 2723,
            "range": "± 31",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_usize/add",
            "value": 3167956,
            "range": "± 7594",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_usize/from_f64",
            "value": 328191,
            "range": "± 2956",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_usize/mul",
            "value": 3515962,
            "range": "± 20865",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_usize/sqrt",
            "value": 232564403,
            "range": "± 2861363",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_usize/to_f64",
            "value": 1118445,
            "range": "± 4529",
            "unit": "ns/iter"
          }
        ]
      }
    ]
  }
}