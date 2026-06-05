window.BENCHMARK_DATA = {
  "lastUpdate": 1780654596502,
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
      }
    ]
  }
}