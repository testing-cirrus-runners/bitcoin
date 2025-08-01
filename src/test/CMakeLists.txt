# Copyright (c) 2023-present The Bitcoin Core developers
# Distributed under the MIT software license, see the accompanying
# file COPYING or https://opensource.org/license/mit/.

# Do not use generator expressions in test sources because the
# SOURCES property is processed to gather test suite macros.

option(PRIORITIZE_LONG_TESTS "Set COST properties to run long tests first (useful for valgrind)" OFF)
add_executable(test_bitcoin
  main.cpp
  addrman_tests.cpp
  allocator_tests.cpp
  amount_tests.cpp
  argsman_tests.cpp
  arith_uint256_tests.cpp
  banman_tests.cpp
  base32_tests.cpp
  base58_tests.cpp
  base64_tests.cpp
  bech32_tests.cpp
  bip32_tests.cpp
  bip324_tests.cpp
  blockchain_tests.cpp
  blockencodings_tests.cpp
  blockfilter_index_tests.cpp
  blockfilter_tests.cpp
  blockmanager_tests.cpp
  bloom_tests.cpp
  bswap_tests.cpp
  chainstate_write_tests.cpp
  checkqueue_tests.cpp
  cluster_linearize_tests.cpp
  coins_tests.cpp
  coinscachepair_tests.cpp
  coinstatsindex_tests.cpp
  common_url_tests.cpp
  compilerbug_tests.cpp
  compress_tests.cpp
  crypto_tests.cpp
  cuckoocache_tests.cpp
  dbwrapper_tests.cpp
  denialofservice_tests.cpp
  descriptor_tests.cpp
  disconnected_transactions.cpp
  feefrac_tests.cpp
  flatfile_tests.cpp
  fs_tests.cpp
  getarg_tests.cpp
  hash_tests.cpp
  headers_sync_chainwork_tests.cpp
  httpserver_tests.cpp
  i2p_tests.cpp
  interfaces_tests.cpp
  key_io_tests.cpp
  key_tests.cpp
  logging_tests.cpp
  mempool_tests.cpp
  merkle_tests.cpp
  merkleblock_tests.cpp
  miner_tests.cpp
  miniminer_tests.cpp
  miniscript_tests.cpp
  minisketch_tests.cpp
  multisig_tests.cpp
  net_peer_connection_tests.cpp
  net_peer_eviction_tests.cpp
  net_tests.cpp
  netbase_tests.cpp
  node_warnings_tests.cpp
  orphanage_tests.cpp
  pcp_tests.cpp
  peerman_tests.cpp
  pmt_tests.cpp
  policy_fee_tests.cpp
  policyestimator_tests.cpp
  pool_tests.cpp
  pow_tests.cpp
  prevector_tests.cpp
  raii_event_tests.cpp
  random_tests.cpp
  rbf_tests.cpp
  rest_tests.cpp
  result_tests.cpp
  reverselock_tests.cpp
  rpc_tests.cpp
  sanity_tests.cpp
  scheduler_tests.cpp
  script_assets_tests.cpp
  script_p2sh_tests.cpp
  script_parse_tests.cpp
  script_segwit_tests.cpp
  script_standard_tests.cpp
  script_tests.cpp
  scriptnum_tests.cpp
  serfloat_tests.cpp
  serialize_tests.cpp
  settings_tests.cpp
  sighash_tests.cpp
  sigopcount_tests.cpp
  skiplist_tests.cpp
  sock_tests.cpp
  span_tests.cpp
  streams_tests.cpp
  sync_tests.cpp
  system_tests.cpp
  testnet4_miner_tests.cpp
  timeoffsets_tests.cpp
  torcontrol_tests.cpp
  transaction_tests.cpp
  translation_tests.cpp
  txdownload_tests.cpp
  txgraph_tests.cpp
  txindex_tests.cpp
  txpackage_tests.cpp
  txreconciliation_tests.cpp
  txrequest_tests.cpp
  txvalidation_tests.cpp
  txvalidationcache_tests.cpp
  uint256_tests.cpp
  util_string_tests.cpp
  util_tests.cpp
  util_threadnames_tests.cpp
  util_trace_tests.cpp
  validation_block_tests.cpp
  validation_chainstate_tests.cpp
  validation_chainstatemanager_tests.cpp
  validation_flush_tests.cpp
  validation_tests.cpp
  validationinterface_tests.cpp
  versionbits_tests.cpp
)

include(TargetDataSources)
target_json_data_sources(test_bitcoin
  data/base58_encode_decode.json
  data/bip341_wallet_vectors.json
  data/blockfilters.json
  data/key_io_invalid.json
  data/key_io_valid.json
  data/script_tests.json
  data/sighash.json
  data/tx_invalid.json
  data/tx_valid.json
)
target_raw_data_sources(test_bitcoin NAMESPACE test::data
  data/asmap.raw
)

add_windows_application_manifest(test_bitcoin)

target_link_libraries(test_bitcoin
  core_interface
  test_util
  bitcoin_cli
  bitcoin_node
  bitcoin_consensus
  minisketch
  secp256k1
  Boost::headers
  libevent::extra
)

if(ENABLE_WALLET)
  add_subdirectory(${PROJECT_SOURCE_DIR}/src/wallet/test wallet)
endif()

if(ENABLE_IPC)
  target_link_libraries(bitcoin_ipc_test
    PRIVATE
      core_interface
      univalue
      Boost::headers
  )

  target_sources(test_bitcoin
    PRIVATE
      ipc_tests.cpp
  )
  target_link_libraries(test_bitcoin bitcoin_ipc_test bitcoin_ipc)
endif()

function(add_boost_test source_file)
  if(NOT EXISTS ${source_file})
    return()
  endif()

  file(READ "${source_file}" source_file_content)
  string(REGEX
    MATCHALL "(BOOST_FIXTURE_TEST_SUITE|BOOST_AUTO_TEST_SUITE)\\(([A-Za-z0-9_]+)"
    test_suite_macro "${source_file_content}"
  )
  list(TRANSFORM test_suite_macro
    REPLACE "(BOOST_FIXTURE_TEST_SUITE|BOOST_AUTO_TEST_SUITE)\\(" ""
  )
  foreach(test_suite_name IN LISTS test_suite_macro)
    add_test(NAME ${test_suite_name}
      COMMAND test_bitcoin --run_test=${test_suite_name} --catch_system_error=no --log_level=test_suite -- DEBUG_LOG_OUT
    )
    set_property(TEST ${test_suite_name} PROPERTY
      SKIP_REGULAR_EXPRESSION "no test cases matching filter" "skipping script_assets_test"
    )
  endforeach()
endfunction()

function(add_all_test_targets)
  get_target_property(test_source_dir test_bitcoin SOURCE_DIR)
  get_target_property(test_sources test_bitcoin SOURCES)
  foreach(test_source ${test_sources})
    cmake_path(IS_RELATIVE test_source result)
    if(result)
      cmake_path(APPEND test_source_dir ${test_source} OUTPUT_VARIABLE test_source)
    endif()
    add_boost_test(${test_source})
  endforeach()
endfunction()

add_all_test_targets()

# Optionally prioritize long-running tests by setting COST properties
# This is useful for valgrind runs where test execution is much slower
if(PRIORITIZE_LONG_TESTS)
  message(STATUS "Prioritizing long-running tests by setting COST properties")

  # COST values are based on some recorded average runtimes * 100
  # CTest runs tests in descending order of cost when --schedule-random is off

  # Top 5 slowest tests
  set_tests_properties(coinselector_tests PROPERTIES COST 37103)
  set_tests_properties(coins_tests_dbbase PROPERTIES COST 29227)
  set_tests_properties(random_tests PROPERTIES COST 25033)
  set_tests_properties(coins_tests_base PROPERTIES COST 19135)
  set_tests_properties(transaction_tests PROPERTIES COST 16410)

  # Other slow tests (>60 seconds)
  set_tests_properties(coins_tests PROPERTIES COST 14598)
  set_tests_properties(miner_tests PROPERTIES COST 14604)
  set_tests_properties(script_tests PROPERTIES COST 14000)
  set_tests_properties(miniscript_tests PROPERTIES COST 13391)
  set_tests_properties(net_tests PROPERTIES COST 12950)
  set_tests_properties(wallet_tests PROPERTIES COST 11761)
  set_tests_properties(argsman_tests PROPERTIES COST 10183)
  set_tests_properties(crypto_tests PROPERTIES COST 10010)
  set_tests_properties(serfloat_tests PROPERTIES COST 9046)
  set_tests_properties(txrequest_tests PROPERTIES COST 8916)
  set_tests_properties(policyestimator_tests PROPERTIES COST 8474)
  set_tests_properties(txvalidationcache_tests PROPERTIES COST 8232)
  set_tests_properties(merkle_tests PROPERTIES COST 7662)
  set_tests_properties(descriptor_tests PROPERTIES COST 6980)
  set_tests_properties(validation_block_tests PROPERTIES COST 6966)
  set_tests_properties(validation_chainstatemanager_tests PROPERTIES COST 6446)
  set_tests_properties(spend_tests PROPERTIES COST 6153)
  set_tests_properties(skiplist_tests PROPERTIES COST 6016)
endif()

install_binary_component(test_bitcoin)
