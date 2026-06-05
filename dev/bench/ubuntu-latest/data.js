window.BENCHMARK_DATA = {
  "lastUpdate": 1780659088905,
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
      }
    ]
  }
}