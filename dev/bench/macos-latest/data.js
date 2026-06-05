window.BENCHMARK_DATA = {
  "lastUpdate": 1780669288945,
  "repoUrl": "https://github.com/ferranSanchezLlado/flexfloat-rs",
  "entries": {
    "FlexFloat (macos-latest)": [
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
        "date": 1780659079103,
        "tool": "cargo",
        "benches": [
          {
            "name": "agm_pi/BoolBitArray",
            "value": 2059435,
            "range": "± 206440",
            "unit": "ns/iter"
          },
          {
            "name": "agm_pi/f64_baseline",
            "value": 23,
            "range": "± 4",
            "unit": "ns/iter"
          },
          {
            "name": "agm_pi/UsizeBitArray",
            "value": 932685,
            "range": "± 140101",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/append_repeated/1024",
            "value": 38137,
            "range": "± 6259",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/append_repeated/256",
            "value": 35609,
            "range": "± 10653",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/append_repeated/64",
            "value": 18871,
            "range": "± 3919",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bits/1024",
            "value": 12813,
            "range": "± 1724",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bits/256",
            "value": 8799,
            "range": "± 1974",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bits/64",
            "value": 7118,
            "range": "± 1234",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bytes/1024",
            "value": 284434,
            "range": "± 38661",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bytes/256",
            "value": 70906,
            "range": "± 14910",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bytes/64",
            "value": 21185,
            "range": "± 4400",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/get_range/1024",
            "value": 23751,
            "range": "± 6552",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/get_range/256",
            "value": 7932,
            "range": "± 1938",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/get_range/64",
            "value": 7023,
            "range": "± 1168",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/shift_fixed/1024",
            "value": 23898,
            "range": "± 6414",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/shift_fixed/256",
            "value": 11965,
            "range": "± 2486",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/shift_fixed/64",
            "value": 12497,
            "range": "± 2212",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/to_bytes/1024",
            "value": 343787,
            "range": "± 49123",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/to_bytes/256",
            "value": 101291,
            "range": "± 22903",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/to_bytes/64",
            "value": 25436,
            "range": "± 10944",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/append_repeated/1024",
            "value": 17871,
            "range": "± 3206",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/append_repeated/256",
            "value": 19643,
            "range": "± 3885",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/append_repeated/64",
            "value": 19459,
            "range": "± 3619",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bits/1024",
            "value": 341076,
            "range": "± 47378",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bits/256",
            "value": 91975,
            "range": "± 13412",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bits/64",
            "value": 26445,
            "range": "± 4872",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bytes/1024",
            "value": 7252,
            "range": "± 1385",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bytes/256",
            "value": 7605,
            "range": "± 2399",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bytes/64",
            "value": 6564,
            "range": "± 1259",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/get_range/1024",
            "value": 7485,
            "range": "± 1479",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/get_range/256",
            "value": 6853,
            "range": "± 1307",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/get_range/64",
            "value": 6996,
            "range": "± 1901",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/shift_fixed/1024",
            "value": 679022,
            "range": "± 79302",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/shift_fixed/256",
            "value": 164677,
            "range": "± 18565",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/shift_fixed/64",
            "value": 40008,
            "range": "± 7949",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/to_bytes/1024",
            "value": 6976,
            "range": "± 1182",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/to_bytes/256",
            "value": 6853,
            "range": "± 1328",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/to_bytes/64",
            "value": 6529,
            "range": "± 1450",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_bool/add",
            "value": 3998938,
            "range": "± 496857",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_bool/from_f64",
            "value": 541238,
            "range": "± 64792",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_bool/mul",
            "value": 7359916,
            "range": "± 927612",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_bool/sqrt",
            "value": 509332755,
            "range": "± 40725758",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_bool/to_f64",
            "value": 593849,
            "range": "± 95177",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_f64_baseline/add",
            "value": 4178,
            "range": "± 594",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_f64_baseline/from_f64",
            "value": 1958,
            "range": "± 157",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_f64_baseline/mul",
            "value": 4048,
            "range": "± 592",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_f64_baseline/sqrt",
            "value": 3355,
            "range": "± 672",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_f64_baseline/to_f64",
            "value": 2898,
            "range": "± 564",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_usize/add",
            "value": 3661335,
            "range": "± 447263",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_usize/from_f64",
            "value": 305245,
            "range": "± 59639",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_usize/mul",
            "value": 3957244,
            "range": "± 539020",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_usize/sqrt",
            "value": 206164073,
            "range": "± 26499849",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_usize/to_f64",
            "value": 1282665,
            "range": "± 181545",
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
        "date": 1780669287804,
        "tool": "cargo",
        "benches": [
          {
            "name": "agm_pi/BoolBitArray",
            "value": 2299163,
            "range": "± 212138",
            "unit": "ns/iter"
          },
          {
            "name": "agm_pi/f64_baseline",
            "value": 22,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "agm_pi/UsizeBitArray",
            "value": 1006682,
            "range": "± 123094",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/append_repeated/1024",
            "value": 34302,
            "range": "± 4524",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/append_repeated/256",
            "value": 28358,
            "range": "± 3726",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/append_repeated/64",
            "value": 18426,
            "range": "± 2807",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bits/1024",
            "value": 12989,
            "range": "± 2139",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bits/256",
            "value": 7026,
            "range": "± 1049",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bits/64",
            "value": 6976,
            "range": "± 931",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bytes/1024",
            "value": 250313,
            "range": "± 56558",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bytes/256",
            "value": 65106,
            "range": "± 11035",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/from_bytes/64",
            "value": 22603,
            "range": "± 4140",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/get_range/1024",
            "value": 19357,
            "range": "± 3028",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/get_range/256",
            "value": 7059,
            "range": "± 1622",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/get_range/64",
            "value": 6930,
            "range": "± 901",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/shift_fixed/1024",
            "value": 20408,
            "range": "± 3695",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/shift_fixed/256",
            "value": 11071,
            "range": "± 1518",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/shift_fixed/64",
            "value": 11749,
            "range": "± 2091",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/to_bytes/1024",
            "value": 346392,
            "range": "± 60873",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/to_bytes/256",
            "value": 80451,
            "range": "± 8773",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_bool/to_bytes/64",
            "value": 21295,
            "range": "± 3456",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/append_repeated/1024",
            "value": 18847,
            "range": "± 4230",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/append_repeated/256",
            "value": 17322,
            "range": "± 2320",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/append_repeated/64",
            "value": 19318,
            "range": "± 4681",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bits/1024",
            "value": 316686,
            "range": "± 21221",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bits/256",
            "value": 85849,
            "range": "± 14217",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bits/64",
            "value": 26149,
            "range": "± 4135",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bytes/1024",
            "value": 6913,
            "range": "± 1134",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bytes/256",
            "value": 6853,
            "range": "± 1252",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/from_bytes/64",
            "value": 6403,
            "range": "± 1215",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/get_range/1024",
            "value": 7477,
            "range": "± 1466",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/get_range/256",
            "value": 6606,
            "range": "± 1212",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/get_range/64",
            "value": 6292,
            "range": "± 1009",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/shift_fixed/1024",
            "value": 710100,
            "range": "± 65845",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/shift_fixed/256",
            "value": 163258,
            "range": "± 19567",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/shift_fixed/64",
            "value": 37713,
            "range": "± 4803",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/to_bytes/1024",
            "value": 6545,
            "range": "± 933",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/to_bytes/256",
            "value": 7005,
            "range": "± 1131",
            "unit": "ns/iter"
          },
          {
            "name": "bitarray_usize/to_bytes/64",
            "value": 6069,
            "range": "± 827",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_bool/add",
            "value": 3834636,
            "range": "± 352941",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_bool/from_f64",
            "value": 550171,
            "range": "± 62415",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_bool/mul",
            "value": 7086506,
            "range": "± 603457",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_bool/sqrt",
            "value": 502190860,
            "range": "± 36160492",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_bool/to_f64",
            "value": 551182,
            "range": "± 48964",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_f64_baseline/add",
            "value": 3858,
            "range": "± 436",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_f64_baseline/from_f64",
            "value": 2021,
            "range": "± 183",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_f64_baseline/mul",
            "value": 3862,
            "range": "± 264",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_f64_baseline/sqrt",
            "value": 3047,
            "range": "± 256",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_f64_baseline/to_f64",
            "value": 2465,
            "range": "± 295",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_usize/add",
            "value": 3152769,
            "range": "± 328451",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_usize/from_f64",
            "value": 268187,
            "range": "± 30822",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_usize/mul",
            "value": 3218667,
            "range": "± 353640",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_usize/sqrt",
            "value": 201331992,
            "range": "± 16860618",
            "unit": "ns/iter"
          },
          {
            "name": "flexfloat_usize/to_f64",
            "value": 1091648,
            "range": "± 69513",
            "unit": "ns/iter"
          }
        ]
      }
    ]
  }
}