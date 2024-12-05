use std::str::FromStr;
use alloy_consensus::Header;
use alloy_eips::BlockId;
use alloy_primitives::{address, keccak256, map::HashSet, Address, Bytes, B256, U256};
use alloy_rpc_types_eth::{state::{EvmOverrides, StateOverride}, transaction::TransactionRequest, BlockOverrides, Index, TransactionInput};
use alloy_rpc_types_trace::{
    filter::TraceFilter,
    opcode::{BlockOpcodeGas, TransactionOpcodeGas},
    parity::*,
    tracerequest::TraceCallRequest,
};
use async_trait::async_trait;
use jsonrpsee::core::RpcResult;
use reth_chainspec::EthereumHardforks;
use reth_consensus_common::calc::{
    base_block_reward, base_block_reward_pre_merge, block_reward, ommer_reward,
};
use reth_evm::ConfigureEvmEnv;
use reth_provider::{BlockReader, ChainSpecProvider, EvmEnvProvider, StateProviderFactory};
use reth_revm::database::StateProviderDatabase;
use reth_rpc_api::TraceApiServer;
use reth_rpc_eth_api::{helpers::TraceExt, FromEthApiError};
use reth_rpc_eth_types::{error::EthApiError, utils::recover_raw_transaction};
use reth_tasks::pool::BlockingTaskGuard;
use revm::{
    db::{CacheDB, DatabaseCommit},
    primitives::EnvWithHandlerCfg,
};
use revm_inspectors::{
    opcode::OpcodeGasInspector,
    tracing::{parity::populate_state_diff, TracingInspector, TracingInspectorConfig},
};
use std::sync::Arc;
use alloy_primitives::hex::FromHex;
use revm_primitives::Bytecode;
use tokio::sync::{AcquireError, OwnedSemaphorePermit};
use stopwatch::Stopwatch;

/// `trace` API implementation.
///
/// This type provides the functionality for handling `trace` related requests.
pub struct TraceApi<Provider, Eth> {
    inner: Arc<TraceApiInner<Provider, Eth>>,
}

struct Token {
    pub address: Address,
    pub storage: U256,
    pub value: U256,
}

// === impl TraceApi ===

impl<Provider, Eth> TraceApi<Provider, Eth> {
    /// The provider that can interact with the chain.
    pub fn provider(&self) -> &Provider {
        &self.inner.provider
    }

    /// Create a new instance of the [`TraceApi`]
    pub fn new(provider: Provider, eth_api: Eth, blocking_task_guard: BlockingTaskGuard) -> Self {
        let inner = Arc::new(TraceApiInner { provider, eth_api, blocking_task_guard });
        Self { inner }
    }

    /// Acquires a permit to execute a tracing call.
    async fn acquire_trace_permit(
        &self,
    ) -> std::result::Result<OwnedSemaphorePermit, AcquireError> {
        self.inner.blocking_task_guard.clone().acquire_owned().await
    }

    /// Access the underlying `Eth` API.
    pub fn eth_api(&self) -> &Eth {
        &self.inner.eth_api
    }
}

// === impl TraceApi ===

impl<Provider, Eth> TraceApi<Provider, Eth>
where
    Provider: BlockReader<Block = <Eth::Provider as BlockReader>::Block>
        + StateProviderFactory
        + EvmEnvProvider
        + ChainSpecProvider<ChainSpec: EthereumHardforks>
        + 'static,
    Eth: TraceExt + 'static,
{
    /// Executes the given call and returns a number of possible traces for it.
    pub async fn trace_call(
        &self,
        trace_request: TraceCallRequest,
    ) -> Result<TraceResults, Eth::Error> {
        let at = trace_request.block_id.unwrap_or_default();
        let config = TracingInspectorConfig::from_parity_config(&trace_request.trace_types);
        let overrides =
            EvmOverrides::new(trace_request.state_overrides, trace_request.block_overrides);
        let mut inspector = TracingInspector::new(config);
        let this = self.clone();
        self.eth_api()
            .spawn_with_call_at(trace_request.call, at, overrides, move |db, env| {
                // wrapper is hack to get around 'higher-ranked lifetime error', see
                // <https://github.com/rust-lang/rust/issues/100013>
                let db = db.0;

                let (res, _) = this.eth_api().inspect(&mut *db, env, &mut inspector)?;
                let trace_res = inspector
                    .into_parity_builder()
                    .into_trace_results_with_state(&res, &trace_request.trace_types, &db)
                    .map_err(Eth::Error::from_eth_err)?;
                Ok(trace_res)
            })
            .await
    }

    /// Traces a call to `eth_sendRawTransaction` without making the call, returning the traces.
    pub async fn trace_raw_transaction(
        &self,
        tx: Bytes,
        trace_types: HashSet<TraceType>,
        block_id: Option<BlockId>,
    ) -> Result<TraceResults, Eth::Error> {
        let tx = recover_raw_transaction(tx)?.into_ecrecovered_transaction();

        let (cfg, block, at) = self.eth_api().evm_env_at(block_id.unwrap_or_default()).await?;

        let env = EnvWithHandlerCfg::new_with_cfg_env(
            cfg,
            block,
            self.eth_api().evm_config().tx_env(tx.as_signed(), tx.signer()),
        );

        let config = TracingInspectorConfig::from_parity_config(&trace_types);

        self.eth_api()
            .spawn_trace_at_with_state(env, config, at, move |inspector, res, db| {
                inspector
                    .into_parity_builder()
                    .into_trace_results_with_state(&res, &trace_types, &db)
                    .map_err(Eth::Error::from_eth_err)
            })
            .await
    }

    /// Performs multiple call traces on top of the same block. i.e. transaction n will be executed
    /// on top of a pending block with all n-1 transactions applied (traced) first.
    ///
    /// Note: Allows tracing dependent transactions, hence all transactions are traced in sequence
    pub async fn trace_call_many(
        &self,
        calls: Vec<(TransactionRequest, HashSet<TraceType>)>,
        block_id: Option<BlockId>,
    ) -> Result<Vec<TraceResults>, Eth::Error> {
        let at = block_id.unwrap_or(BlockId::pending());
        let (cfg, block_env, at) = self.eth_api().evm_env_at(at).await?;

        let this = self.clone();
        // execute all transactions on top of each other and record the traces
        self.eth_api()
            .spawn_with_state_at_block(at, move |state| {
                let mut results = Vec::with_capacity(calls.len());
                let mut db = CacheDB::new(StateProviderDatabase::new(state));

                let mut calls = calls.into_iter().peekable();

                while let Some((call, trace_types)) = calls.next() {
                    let env = this.eth_api().prepare_call_env(
                        cfg.clone(),
                        block_env.clone(),
                        call,
                        &mut db,
                        Default::default(),
                    )?;
                    let config = TracingInspectorConfig::from_parity_config(&trace_types);
                    let mut inspector = TracingInspector::new(config);
                    let (res, _) = this.eth_api().inspect(&mut db, env, &mut inspector)?;

                    let trace_res = inspector
                        .into_parity_builder()
                        .into_trace_results_with_state(&res, &trace_types, &db)
                        .map_err(Eth::Error::from_eth_err)?;

                    results.push(trace_res);

                    // need to apply the state changes of this call before executing the
                    // next call
                    if calls.peek().is_some() {
                        // need to apply the state changes of this call before executing
                        // the next call
                        db.commit(res.state)
                    }
                }

                Ok(results)
            })
            .await
    }

    /// Replays a transaction, returning the traces.
    pub async fn replay_transaction(
        &self,
        hash: B256,
        trace_types: HashSet<TraceType>,
    ) -> Result<TraceResults, Eth::Error> {
        let config = TracingInspectorConfig::from_parity_config(&trace_types);
        self.eth_api()
            .spawn_trace_transaction_in_block(hash, config, move |_, inspector, res, db| {
                let trace_res = inspector
                    .into_parity_builder()
                    .into_trace_results_with_state(&res, &trace_types, &db)
                    .map_err(Eth::Error::from_eth_err)?;
                Ok(trace_res)
            })
            .await
            .transpose()
            .ok_or(EthApiError::TransactionNotFound)?
    }

    /// Returns transaction trace objects at the given index
    ///
    /// Note: For compatibility reasons this only supports 1 single index, since this method is
    /// supposed to return a single trace. See also: <https://github.com/ledgerwatch/erigon/blob/862faf054b8a0fa15962a9c73839b619886101eb/turbo/jsonrpc/trace_filtering.go#L114-L133>
    ///
    /// This returns `None` if `indices` is empty
    pub async fn trace_get(
        &self,
        hash: B256,
        indices: Vec<usize>,
    ) -> Result<Option<LocalizedTransactionTrace>, Eth::Error> {
        if indices.len() != 1 {
            // The OG impl failed if it gets more than a single index
            return Ok(None)
        }
        self.trace_get_index(hash, indices[0]).await
    }

    /// Returns transaction trace object at the given index.
    ///
    /// Returns `None` if the trace object at that index does not exist
    pub async fn trace_get_index(
        &self,
        hash: B256,
        index: usize,
    ) -> Result<Option<LocalizedTransactionTrace>, Eth::Error> {
        Ok(self.trace_transaction(hash).await?.and_then(|traces| traces.into_iter().nth(index)))
    }

    /// Returns all transaction traces that match the given filter.
    ///
    /// This is similar to [`Self::trace_block`] but only returns traces for transactions that match
    /// the filter.
    pub async fn trace_filter(
        &self,
        filter: TraceFilter,
    ) -> Result<Vec<LocalizedTransactionTrace>, Eth::Error> {
        // We'll reuse the matcher across multiple blocks that are traced in parallel
        let matcher = Arc::new(filter.matcher());
        let TraceFilter { from_block, to_block, after, count, .. } = filter;
        let start = from_block.unwrap_or(0);
        let end = if let Some(to_block) = to_block {
            to_block
        } else {
            self.provider().best_block_number().map_err(Eth::Error::from_eth_err)?
        };

        if start > end {
            return Err(EthApiError::InvalidParams(
                "invalid parameters: fromBlock cannot be greater than toBlock".to_string(),
            )
            .into())
        }

        // ensure that the range is not too large, since we need to fetch all blocks in the range
        let distance = end.saturating_sub(start);
        if distance > 100 {
            return Err(EthApiError::InvalidParams(
                "Block range too large; currently limited to 100 blocks".to_string(),
            )
            .into())
        }

        // fetch all blocks in that range
        let blocks = self
            .provider()
            .sealed_block_with_senders_range(start..=end)
            .map_err(Eth::Error::from_eth_err)?
            .into_iter()
            .map(Arc::new)
            .collect::<Vec<_>>();

        // trace all blocks
        let mut block_traces = Vec::with_capacity(blocks.len());
        for block in &blocks {
            let matcher = matcher.clone();
            let traces = self.eth_api().trace_block_until(
                block.hash().into(),
                Some(block.clone()),
                None,
                TracingInspectorConfig::default_parity(),
                move |tx_info, inspector, _, _, _| {
                    let mut traces =
                        inspector.into_parity_builder().into_localized_transaction_traces(tx_info);
                    traces.retain(|trace| matcher.matches(&trace.trace));
                    Ok(Some(traces))
                },
            );
            block_traces.push(traces);
        }

        let block_traces = futures::future::try_join_all(block_traces).await?;
        let mut all_traces = block_traces
            .into_iter()
            .flatten()
            .flat_map(|traces| traces.into_iter().flatten().flat_map(|traces| traces.into_iter()))
            .collect::<Vec<_>>();

        // add reward traces for all blocks
        for block in &blocks {
            if let Some(base_block_reward) = self.calculate_base_block_reward(&block.header)? {
                all_traces.extend(
                    self.extract_reward_traces(
                        &block.header,
                        &block.body.ommers,
                        base_block_reward,
                    )
                    .into_iter()
                    .filter(|trace| matcher.matches(&trace.trace)),
                );
            } else {
                // no block reward, means we're past the Paris hardfork and don't expect any rewards
                // because the blocks in ascending order
                break
            }
        }

        // Skips the first `after` number of matching traces.
        // If `after` is greater than or equal to the number of matched traces, it returns an empty
        // array.
        if let Some(after) = after.map(|a| a as usize) {
            if after < all_traces.len() {
                all_traces.drain(..after);
            } else {
                return Ok(vec![])
            }
        }

        // Return at most `count` of traces
        if let Some(count) = count {
            let count = count as usize;
            if count < all_traces.len() {
                all_traces.truncate(count);
            }
        };

        Ok(all_traces)
    }

    /// Returns all traces for the given transaction hash
    pub async fn trace_transaction(
        &self,
        hash: B256,
    ) -> Result<Option<Vec<LocalizedTransactionTrace>>, Eth::Error> {
        self.eth_api()
            .spawn_trace_transaction_in_block(
                hash,
                TracingInspectorConfig::default_parity(),
                move |tx_info, inspector, _, _| {
                    let traces =
                        inspector.into_parity_builder().into_localized_transaction_traces(tx_info);
                    Ok(traces)
                },
            )
            .await
    }

    /// Returns traces created at given block.
    pub async fn trace_block(
        &self,
        block_id: BlockId,
    ) -> Result<Option<Vec<LocalizedTransactionTrace>>, Eth::Error> {
        let traces = self.eth_api().trace_block_with(
            block_id,
            None,
            TracingInspectorConfig::default_parity(),
            |tx_info, inspector, _, _, _| {
                let traces =
                    inspector.into_parity_builder().into_localized_transaction_traces(tx_info);
                Ok(traces)
            },
        );

        let block = self.eth_api().block_with_senders(block_id);
        let (maybe_traces, maybe_block) = futures::try_join!(traces, block)?;

        let mut maybe_traces =
            maybe_traces.map(|traces| traces.into_iter().flatten().collect::<Vec<_>>());

        if let (Some(block), Some(traces)) = (maybe_block, maybe_traces.as_mut()) {
            if let Some(base_block_reward) = self.calculate_base_block_reward(&block.header)? {
                traces.extend(self.extract_reward_traces(
                    &block.header,
                    &block.body.ommers,
                    base_block_reward,
                ));
            }
        }

        Ok(maybe_traces)
    }

    /// Replays all transactions in a block
    pub async fn replay_block_transactions(
        &self,
        block_id: BlockId,
        trace_types: HashSet<TraceType>,
    ) -> Result<Option<Vec<TraceResultsWithTransactionHash>>, Eth::Error> {
        self.eth_api()
            .trace_block_with(
                block_id,
                None,
                TracingInspectorConfig::from_parity_config(&trace_types),
                move |tx_info, inspector, res, state, db| {
                    let mut full_trace =
                        inspector.into_parity_builder().into_trace_results(&res, &trace_types);

                    // If statediffs were requested, populate them with the account balance and
                    // nonce from pre-state
                    if let Some(ref mut state_diff) = full_trace.state_diff {
                        populate_state_diff(state_diff, db, state.iter())
                            .map_err(Eth::Error::from_eth_err)?;
                    }

                    let trace = TraceResultsWithTransactionHash {
                        transaction_hash: tx_info.hash.expect("tx hash is set"),
                        full_trace,
                    };
                    Ok(trace)
                },
            )
            .await
    }

    /// Returns all opcodes with their count and combined gas usage for the given transaction in no
    /// particular order.
    pub async fn trace_transaction_opcode_gas(
        &self,
        tx_hash: B256,
    ) -> Result<Option<TransactionOpcodeGas>, Eth::Error> {
        self.eth_api()
            .spawn_trace_transaction_in_block_with_inspector(
                tx_hash,
                OpcodeGasInspector::default(),
                move |_tx_info, inspector, _res, _| {
                    let trace = TransactionOpcodeGas {
                        transaction_hash: tx_hash,
                        opcode_gas: inspector.opcode_gas_iter().collect(),
                    };
                    Ok(trace)
                },
            )
            .await
    }

    /// Returns the opcodes of all transactions in the given block.
    ///
    /// This is the same as [`Self::trace_transaction_opcode_gas`] but for all transactions in a
    /// block.
    pub async fn trace_block_opcode_gas(
        &self,
        block_id: BlockId,
    ) -> Result<Option<BlockOpcodeGas>, Eth::Error> {
        let res = self
            .eth_api()
            .trace_block_inspector(
                block_id,
                None,
                OpcodeGasInspector::default,
                move |tx_info, inspector, _res, _, _| {
                    let trace = TransactionOpcodeGas {
                        transaction_hash: tx_info.hash.expect("tx hash is set"),
                        opcode_gas: inspector.opcode_gas_iter().collect(),
                    };
                    Ok(trace)
                },
            )
            .await?;

        let Some(transactions) = res else { return Ok(None) };

        let Some(block) = self.eth_api().block_with_senders(block_id).await? else {
            return Ok(None)
        };

        Ok(Some(BlockOpcodeGas {
            block_hash: block.hash(),
            block_number: block.header.number,
            transactions,
        }))
    }

    /// Calculates the base block reward for the given block:
    ///
    /// - if Paris hardfork is activated, no block rewards are given
    /// - if Paris hardfork is not activated, calculate block rewards with block number only
    /// - if Paris hardfork is unknown, calculate block rewards with block number and ttd
    fn calculate_base_block_reward(&self, header: &Header) -> Result<Option<u128>, Eth::Error> {
        let chain_spec = self.provider().chain_spec();
        let is_paris_activated = chain_spec.is_paris_active_at_block(header.number);

        Ok(match is_paris_activated {
            Some(true) => None,
            Some(false) => Some(base_block_reward_pre_merge(&chain_spec, header.number)),
            None => {
                // if Paris hardfork is unknown, we need to fetch the total difficulty at the
                // block's height and check if it is pre-merge to calculate the base block reward
                if let Some(header_td) = self
                    .provider()
                    .header_td_by_number(header.number)
                    .map_err(Eth::Error::from_eth_err)?
                {
                    base_block_reward(
                        chain_spec.as_ref(),
                        header.number,
                        header.difficulty,
                        header_td,
                    )
                } else {
                    None
                }
            }
        })
    }

    /// Extracts the reward traces for the given block:
    ///  - block reward
    ///  - uncle rewards
    fn extract_reward_traces(
        &self,
        header: &Header,
        ommers: &[Header],
        base_block_reward: u128,
    ) -> Vec<LocalizedTransactionTrace> {
        let mut traces = Vec::with_capacity(ommers.len() + 1);

        let block_reward = block_reward(base_block_reward, ommers.len());
        traces.push(reward_trace(
            header,
            RewardAction {
                author: header.beneficiary,
                reward_type: RewardType::Block,
                value: U256::from(block_reward),
            },
        ));

        for uncle in ommers {
            let uncle_reward = ommer_reward(base_block_reward, header.number, uncle.number);
            traces.push(reward_trace(
                header,
                RewardAction {
                    author: uncle.beneficiary,
                    reward_type: RewardType::Uncle,
                    value: U256::from(uncle_reward),
                },
            ));
        }
        traces
    }

    /// Performs multiple call traces on top of the same block. i.e. transaction n will be executed
    /// on top of a pending block with all n-1 transactions applied (traced) first.
    ///
    /// Note: Allows tracing dependent transactions, hence all transactions are traced in sequence
    pub async fn trace_call_many_custom(
        &self,
        calls: Vec<(TransactionRequest, HashSet<TraceType>)>,
        block_id: Option<BlockId>,
    ) -> Result<Vec<TraceResults>, Eth::Error> {
        let at = block_id.unwrap_or(BlockId::pending());
        let (cfg, block_env, at) = self.eth_api().evm_env_at(at).await?;

        let this = self.clone();
        // execute all transactions on top of each other and record the traces
        self.eth_api()
            .spawn_with_state_at_block(at, move |state| {
                let mut results = Vec::with_capacity(calls.len());
                let mut db = CacheDB::new(StateProviderDatabase::new(state));

                let mut stopwatch = Stopwatch::start_new();
                let tokens: [Token; 4] = [
                    Token {
                        address: address!("c02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"),
                        storage: U256::from(3),
                        value: U256::from_str("1_000_000_000_000_000_000_000").unwrap(),
                    }, // weth
                    Token {
                        address: address!("6b175474e89094c44da98b954eedeac495271d0f"),
                        storage: U256::from(2),
                        value: U256::from_str("1_000_000_000_000_000_000_000_000").unwrap(),
                    }, // dai
                    Token {
                        address: address!("a0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"),
                        storage: U256::from(9),
                        value: U256::from_str("100_000_000_000_000").unwrap(),
                    }, // usdc
                    Token {
                        address: address!("dac17f958d2ee523a2206206994597c13d831ec7"),
                        storage: U256::from(2),
                        value: U256::from_str("100_000_000_000_000").unwrap(),
                    }, // usdt
                ];

                for item in tokens.iter().enumerate() {
                    let token = item.1;
                    let contract = format!("{:0>64}", "Bd770416a3345F91E4B34576cb804a576fa48EB1");
                    let storage = format!("{:0>64}", token.storage);
                    let encode_str = contract + storage.as_str();
                    let encode = Bytes::from_str(encode_str.as_str()).unwrap();
                    let hashed_acc_balance_slot = keccak256(encode);
                    db.insert_account_storage(token.address, hashed_acc_balance_slot.into(),token.value).unwrap();
                }

                let contract_address = Address::from_str("0xBd770416a3345F91E4B34576cb804a576fa48EB1").unwrap();

                let db_account = db.load_account(contract_address).unwrap();
                let acc_info = &mut db_account.info.clone();

                acc_info.code = Some(Bytecode::LegacyRaw(
                    Bytes::from_str("6080604052600436106100ea575f3560e01c8063a224ef8311610083578063f40a74a811610055578063f40a74a8146100f3578063f46420d514610227578063fa461e33146100f3578063fa483e72146100f357005b8063a224ef83146100f3578063ad49bc27146101e9578063b134ef53146100f3578063e1f21c671461020857005b8063359ce5cd116100bc578063359ce5cd1461017b5780633a1c453c146100f35780634aa4a4fc1461019d57806388e5d910146100f357005b80632175df8d146100f357806323a69e75146100f35780632d9802f414610112578063327494611461014757005b366100f157005b005b3480156100fe575f80fd5b506100f161010d366004611ffe565b610253565b34801561011d575f80fd5b5061013161012c36600461209f565b61029c565b60405161013e919061211d565b60405180910390f35b348015610152575f80fd5b50610166610161366004612134565b6107fb565b6040805192835260208301919091520161013e565b348015610186575f80fd5b5061018f600381565b60405190815260200161013e565b3480156101a8575f80fd5b506101c473c02aaa39b223fe8d0a0e5c4f27ead9083c756cc281565b60405173ffffffffffffffffffffffffffffffffffffffff909116815260200161013e565b3480156101f4575f80fd5b5061018f610203366004612184565b6108eb565b348015610213575f80fd5b506100f1610222366004612202565b610b14565b348015610232575f80fd5b5061024661024136600461223b565b610b24565b60405161013e91906122b4565b5f610260828401846123af565b90505f80610270835f0151610d43565b915091505f80881315610284575086610287565b50855b610292833383610d63565b5050505050505050565b604080518082019091525f80825260208201525f80806102c26060880160408901612496565b61036c576102d7610100880160e089016124b8565b6040517f70a0823100000000000000000000000000000000000000000000000000000000815273ffffffffffffffffffffffffffffffffffffffff888116600483015291909116906370a0823190602401602060405180830381865afa158015610343573d5f803e3d5ffd5b505050506040513d601f19601f8201168201806040525081019061036791906124d1565b61040c565b61037c60e0880160c089016124b8565b6040517f70a0823100000000000000000000000000000000000000000000000000000000815273ffffffffffffffffffffffffffffffffffffffff888116600483015291909116906370a0823190602401602060405180830381865afa1580156103e8573d5f803e3d5ffd5b505050506040513d601f19601f8201168201806040525081019061040c91906124d1565b9050600361041d6020890189612515565b600581111561042e5761042e6124e8565b036104965761048f8861044760408a0160208b016124b8565b61045760a08b0160808c01612496565b61046760e08c0160c08d016124b8565b6104786101008d0160e08e016124b8565b61048860608e0160408f01612496565b8c8c610eb4565b915061060d565b60026104a56020890189612515565b60058111156104b6576104b66124e8565b036105165761048f886104cf60408a0160208b016124b8565b6104df60808b0160608c01612496565b6104ef60e08c0160c08d016124b8565b6105006101008d0160e08e016124b8565b61051060608e0160408f01612496565b8c6108eb565b5f6105246020890189612515565b6005811115610535576105356124e8565b0361058b5761048f8861054e60408a0160208b016124b8565b61055e60e08b0160c08c016124b8565b61056f6101008c0160e08d016124b8565b6101008c013561058560608e0160408f01612496565b8c611104565b600161059a6020890189612515565b60058111156105ab576105ab6124e8565b0361060d5761060a886105c460408a0160208b016124b8565b6105d460e08b0160c08c016124b8565b6105e56101008c0160e08d016124b8565b8b61012001358c61014001358d60a00160208101906106049190612533565b8d6113c4565b91505b5f61061e6060890160408a01612496565b6106c857610633610100890160e08a016124b8565b6040517f70a0823100000000000000000000000000000000000000000000000000000000815273ffffffffffffffffffffffffffffffffffffffff898116600483015291909116906370a0823190602401602060405180830381865afa15801561069f573d5f803e3d5ffd5b505050506040513d601f19601f820116820180604052508101906106c391906124d1565b610768565b6106d860e0890160c08a016124b8565b6040517f70a0823100000000000000000000000000000000000000000000000000000000815273ffffffffffffffffffffffffffffffffffffffff898116600483015291909116906370a0823190602401602060405180830381865afa158015610744573d5f803e3d5ffd5b505050506040513d601f19601f8201168201806040525081019061076891906124d1565b905061077a6060890160408a01612496565b61078d57610788828261259f565b610797565b610797818361259f565b93506107a96060890160408a01612496565b80156107d2575060026107bf60208a018a612515565b60058111156107d0576107d06124e8565b145b156107db578392505b505060408051808201909152908152602081019190915295945050505050565b5f805f6108088585611622565b5090505f808773ffffffffffffffffffffffffffffffffffffffff16630902f1ac6040518163ffffffff1660e01b8152600401606060405180830381865afa158015610856573d5f803e3d5ffd5b505050506040513d601f19601f8201168201806040525081019061087a91906125cf565b506dffffffffffffffffffffffffffff1691506dffffffffffffffffffffffffffff1691508273ffffffffffffffffffffffffffffffffffffffff168773ffffffffffffffffffffffffffffffffffffffff16146108d95780826108dc565b81815b90999098509650505050505050565b6040805180820182527fffffffffffffffffffffffffffffffffffffffff000000000000000000000000606087811b82168184015286901b166074820152815180820360680181526088820190925290815273ffffffffffffffffffffffffffffffffffffffff821660208201525f9081871561096e57506401000276a4610985565b5073fffd8963efd1fc6a506488495d951d5263988d255b5f808a73ffffffffffffffffffffffffffffffffffffffff1663128acb08878c8a6109b0578f6109ba565b8f6109ba9061261b565b87896040516020016109cc91906126bc565b6040516020818303038152906040526040518663ffffffff1660e01b81526004016109fb959493929190612703565b60408051808303815f875af1158015610a16573d5f803e3d5ffd5b505050506040513d601f19601f82011682018060405250810190610a3a9190612749565b915091505f8a610a4a5782610a4c565b815b610a559061261b565b9050878015610a6457508c8114155b15610b025760405162461bcd60e51b815260206004820152604360248201527f596f752077696c6c2072656365697665206665776572206f72206d6f7265206460448201527f657374696e6174696f6e20746f6b656e207468616e20796f757220657870656360648201527f7465640000000000000000000000000000000000000000000000000000000000608482015260a4015b60405180910390fd5b9450505050505b979650505050505050565b610b1f838383611771565b505050565b60605f82818167ffffffffffffffff811115610b4257610b4261230a565b604051908082528060200260200182016040528015610b8657816020015b604080518082019091525f8082526020820152815260200190600190039081610b605790505b5090505f5b828160ff161015610c9c57610ba160018461259f565b8160ff1614158015610bf9575060038787610bbd84600161276b565b60ff16818110610bcf57610bcf612784565b610be6926020610160909202019081019150612515565b6005811115610bf757610bf76124e8565b145b15610c3d578686610c0b83600161276b565b60ff16818110610c1d57610c1d612784565b905061016002016020016020810190610c3691906124b8565b9350610c41565b3093505b5f610c698989898560ff16818110610c5b57610c5b612784565b90506101600201878561029c565b90508060200151985080838360ff1681518110610c8857610c88612784565b602090810291909101015250600101610b8b565b5080610ca960018461259f565b81518110610cb957610cb9612784565b6020026020010151602001515f03610d395760405162461bcd60e51b815260206004820152602960248201527f546865206c6173742073746570206f66207468652072657475726e2076616c7560448201527f65206973207a65726f00000000000000000000000000000000000000000000006064820152608401610af9565b9695505050505050565b5f80610d4f835f6118bb565b9150610d5c8360146118bb565b9050915091565b6040805173ffffffffffffffffffffffffffffffffffffffff8481166024830152604480830185905283518084039091018152606490920183526020820180517bffffffffffffffffffffffffffffffffffffffffffffffffffffffff167fa9059cbb0000000000000000000000000000000000000000000000000000000017905291515f92839290871691610df991906127b1565b5f604051808303815f865af19150503d805f8114610e32576040519150601f19603f3d011682016040523d82523d5f602084013e610e37565b606091505b5091509150818015610e61575080511580610e61575080806020019051810190610e6191906127cc565b610ead5760405162461bcd60e51b815260206004820152600260248201527f53540000000000000000000000000000000000000000000000000000000000006044820152606401610af9565b5050505050565b6040805160028082526060820183525f928392919060208301908036833701905050905086815f81518110610eeb57610eeb612784565b602002602001019073ffffffffffffffffffffffffffffffffffffffff16908173ffffffffffffffffffffffffffffffffffffffff16815250508581600181518110610f3957610f39612784565b602002602001019073ffffffffffffffffffffffffffffffffffffffff16908173ffffffffffffffffffffffffffffffffffffffff168152505060608515610fb157610f868a8c84611990565b9050610fac888b835f81518110610f9f57610f9f612784565b6020026020010151610d63565b610fd4565b8360ff165f03610fc657610fc6888b8d610d63565b610fd18a8c84611bed565b90505b60605f8a610fe2575f610ffe565b82600181518110610ff557610ff5612784565b60200260200101515b90505f8b611026578360018151811061101957611019612784565b6020026020010151611028565b5f5b6040517f022c0d9f00000000000000000000000000000000000000000000000000000000815290915073ffffffffffffffffffffffffffffffffffffffff8e169063022c0d9f9061108390859085908d9089906004016127e7565b5f604051808303815f87803b15801561109a575f80fd5b505af11580156110ac573d5f803e3d5ffd5b50505050886110d557836001815181106110c8576110c8612784565b60200260200101516110f0565b835f815181106110e7576110e7612784565b60200260200101515b955050505050505b98975050505050505050565b6040517f70a082310000000000000000000000000000000000000000000000000000000081523060048201525f90819073ffffffffffffffffffffffffffffffffffffffff8816906370a0823190602401602060405180830381865afa158015611170573d5f803e3d5ffd5b505050506040513d601f19601f8201168201806040525081019061119491906124d1565b90506111b58773ba12222222228d8ba445958a75a0704d566bf2c883610b14565b5f6040518060c00160405280878152602001866111d2575f6111d5565b60015b60018111156111e6576111e66124e8565b81526020018973ffffffffffffffffffffffffffffffffffffffff1681526020018873ffffffffffffffffffffffffffffffffffffffff1681526020018b81526020016040518060400160405280600281526020017f307800000000000000000000000000000000000000000000000000000000000081525081525090505f60405180608001604052803073ffffffffffffffffffffffffffffffffffffffff1681526020015f151581526020018673ffffffffffffffffffffffffffffffffffffffff1681526020015f151581525090505f8073ba12222222228d8ba445958a75a0704d566bf2c873ffffffffffffffffffffffffffffffffffffffff166352bbbe2985858b6112f7575f6112f9565b885b63843c2acc6040516024016113119493929190612821565b6040516020818303038152906040529060e01b6020820180517bffffffffffffffffffffffffffffffffffffffffffffffffffffffff838183161783525050505060405161135f91906127b1565b5f604051808303815f865af19150503d805f8114611398576040519150601f19603f3d011682016040523d82523d5f602084013e61139d565b606091505b509150915081156113c0575f6113b28261294d565b9650610b0995505050505050565b5f80fd5b5f8060606113d3898b8d610b14565b7f31829afd000000000000000000000000000000000000000000000000000000007fffffffff0000000000000000000000000000000000000000000000000000000086160161151557604080516024810189905260448101889052606481018d90525f6084820181905260a482015273ffffffffffffffffffffffffffffffffffffffff86811660c4808401919091528351808403909101815260e490920183526020820180517bffffffffffffffffffffffffffffffffffffffffffffffffffffffff167fce7d6503000000000000000000000000000000000000000000000000000000001790529151918c16916114cc91906127b1565b5f604051808303815f865af19150503d805f8114611505576040519150601f19603f3d011682016040523d82523d5f602084013e61150a565b606091505b509092509050611607565b60408051600f89810b602483015288900b6044820152606481018d90525f608482015273ffffffffffffffffffffffffffffffffffffffff86811660a4808401919091528351808403909101815260c490920183526020820180517bffffffffffffffffffffffffffffffffffffffffffffffffffffffff167fddc1f59d000000000000000000000000000000000000000000000000000000001790529151918c16916115c291906127b1565b5f604051808303815f865af19150503d805f81146115fb576040519150601f19603f3d011682016040523d82523d5f602084013e611600565b606091505b5090925090505b81156113c0575f6116178261294d565b93506110f892505050565b5f808273ffffffffffffffffffffffffffffffffffffffff168473ffffffffffffffffffffffffffffffffffffffff16036116c55760405162461bcd60e51b815260206004820152602560248201527f556e697377617056324c6962726172793a204944454e544943414c5f4144445260448201527f45535345530000000000000000000000000000000000000000000000000000006064820152608401610af9565b8273ffffffffffffffffffffffffffffffffffffffff168473ffffffffffffffffffffffffffffffffffffffff16106116ff578284611702565b83835b909250905073ffffffffffffffffffffffffffffffffffffffff821661176a5760405162461bcd60e51b815260206004820152601e60248201527f556e697377617056324c6962726172793a205a45524f5f4144445245535300006044820152606401610af9565b9250929050565b6040805173ffffffffffffffffffffffffffffffffffffffff8481166024830152604480830185905283518084039091018152606490920183526020820180517bffffffffffffffffffffffffffffffffffffffffffffffffffffffff167f095ea7b30000000000000000000000000000000000000000000000000000000017905291515f9283929087169161180791906127b1565b5f604051808303815f865af19150503d805f8114611840576040519150601f19603f3d011682016040523d82523d5f602084013e611845565b606091505b509150915081801561186f57508051158061186f57508080602001905181019061186f91906127cc565b610ead5760405162461bcd60e51b815260206004820152600260248201527f53410000000000000000000000000000000000000000000000000000000000006044820152606401610af9565b5f816118c8816014612992565b10156119165760405162461bcd60e51b815260206004820152601260248201527f746f416464726573735f6f766572666c6f7700000000000000000000000000006044820152606401610af9565b611921826014612992565b835110156119715760405162461bcd60e51b815260206004820152601560248201527f746f416464726573735f6f75744f66426f756e647300000000000000000000006044820152606401610af9565b50818101602001516c0100000000000000000000000090045b92915050565b60606002825110156119e45760405162461bcd60e51b815260206004820152601e60248201527f556e697377617056324c6962726172793a20494e56414c49445f5041544800006044820152606401610af9565b815167ffffffffffffffff8111156119fe576119fe61230a565b604051908082528060200260200182016040528015611a27578160200160208202803683370190505b5090508281600181518110611a3e57611a3e612784565b6020026020010181815250505f80611a8a86855f81518110611a6257611a62612784565b602002602001015186600181518110611a7d57611a7d612784565b60200260200101516107fb565b915091505f8511611b035760405162461bcd60e51b815260206004820152602c60248201527f556e697377617056324c6962726172793a20494e53554646494349454e545f4f60448201527f55545055545f414d4f554e5400000000000000000000000000000000000000006064820152608401610af9565b5f82118015611b1157505f81115b611b835760405162461bcd60e51b815260206004820152602860248201527f556e697377617056324c6962726172793a20494e53554646494349454e545f4c60448201527f49515549444954590000000000000000000000000000000000000000000000006064820152608401610af9565b5f611b9a6103e8611b948589611ed5565b90611ed5565b90505f611bad6103e5611b94858a611f44565b9050611bc46001611bbe83856129a5565b90611fa1565b855f81518110611bd657611bd6612784565b602002602001018181525050505050509392505050565b6060600282511015611c415760405162461bcd60e51b815260206004820152601e60248201527f556e697377617056324c6962726172793a20494e56414c49445f5041544800006044820152606401610af9565b815167ffffffffffffffff811115611c5b57611c5b61230a565b604051908082528060200260200182016040528015611c84578160200160208202803683370190505b50905082815f81518110611c9a57611c9a612784565b6020026020010181815250505f80611cbe86855f81518110611a6257611a62612784565b915091505f82118015611cd057505f81115b611d425760405162461bcd60e51b815260206004820152602860248201527f556e697377617056324c6962726172793a20494e53554646494349454e545f4c60448201527f49515549444954590000000000000000000000000000000000000000000000006064820152608401610af9565b5f611df983865f81518110611d5957611d59612784565b60209081029190910101516040517f70a0823100000000000000000000000000000000000000000000000000000000815273ffffffffffffffffffffffffffffffffffffffff8b81166004830152909116906370a0823190602401602060405180830381865afa158015611dcf573d5f803e3d5ffd5b505050506040513d601f19601f82011682018060405250810190611df391906124d1565b90611f44565b90505f8111611e705760405162461bcd60e51b815260206004820152602b60248201527f556e697377617056324c6962726172793a20494e53554646494349454e545f4960448201527f4e5055545f414d4f554e540000000000000000000000000000000000000000006064820152608401610af9565b5f611e7d826103e5611ed5565b90505f611e8a8285611ed5565b90505f611e9d83611bbe886103e8611ed5565b9050611ea981836129a5565b87600181518110611ebc57611ebc612784565b6020026020010181815250505050505050509392505050565b5f811580611ef857508282611eea81836129dd565b9250611ef690836129a5565b145b61198a5760405162461bcd60e51b815260206004820152601460248201527f64732d6d6174682d6d756c2d6f766572666c6f770000000000000000000000006044820152606401610af9565b5f82611f50838261259f565b915081111561198a5760405162461bcd60e51b815260206004820152601560248201527f64732d6d6174682d7375622d756e646572666c6f7700000000000000000000006044820152606401610af9565b5f82611fad8382612992565b915081101561198a5760405162461bcd60e51b815260206004820152601460248201527f64732d6d6174682d6164642d6f766572666c6f770000000000000000000000006044820152606401610af9565b5f805f8060608587031215612011575f80fd5b8435935060208501359250604085013567ffffffffffffffff80821115612036575f80fd5b818701915087601f830112612049575f80fd5b813581811115612057575f80fd5b886020828501011115612068575f80fd5b95989497505060200194505050565b803573ffffffffffffffffffffffffffffffffffffffff8116811461209a575f80fd5b919050565b5f805f808486036101c08112156120b4575f80fd5b853594506101607fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffe0820112156120e8575f80fd5b506020850192506120fc6101808601612077565b91506101a085013560ff81168114612112575f80fd5b939692955090935050565b81518152602080830151908201526040810161198a565b5f805f60608486031215612146575f80fd5b61214f84612077565b925061215d60208501612077565b915061216b60408501612077565b90509250925092565b8015158114612181575f80fd5b50565b5f805f805f805f60e0888a03121561219a575f80fd5b873596506121aa60208901612077565b955060408801356121ba81612174565b94506121c860608901612077565b93506121d660808901612077565b925060a08801356121e681612174565b91506121f460c08901612077565b905092959891949750929550565b5f805f60608486031215612214575f80fd5b61221d84612077565b925061222b60208501612077565b9150604084013590509250925092565b5f805f6040848603121561224d575f80fd5b83359250602084013567ffffffffffffffff8082111561226b575f80fd5b818601915086601f83011261227e575f80fd5b81358181111561228c575f80fd5b876020610160830285010111156122a1575f80fd5b6020830194508093505050509250925092565b602080825282518282018190525f919060409081850190868401855b828110156122fd576122ed84835180518252602090810151910152565b92840192908501906001016122d0565b5091979650505050505050565b7f4e487b71000000000000000000000000000000000000000000000000000000005f52604160045260245ffd5b6040805190810167ffffffffffffffff8111828210171561235a5761235a61230a565b60405290565b604051601f82017fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffe016810167ffffffffffffffff811182821017156123a7576123a761230a565b604052919050565b5f60208083850312156123c0575f80fd5b823567ffffffffffffffff808211156123d7575f80fd5b90840190604082870312156123ea575f80fd5b6123f2612337565b823582811115612400575f80fd5b8301601f81018813612410575f80fd5b8035838111156124225761242261230a565b612452867fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffe0601f84011601612360565b93508084528886828401011115612467575f80fd5b80868301878601375f90840186015250818152612485838501612077565b848201528094505050505092915050565b5f602082840312156124a6575f80fd5b81356124b181612174565b9392505050565b5f602082840312156124c8575f80fd5b6124b182612077565b5f602082840312156124e1575f80fd5b5051919050565b7f4e487b71000000000000000000000000000000000000000000000000000000005f52602160045260245ffd5b5f60208284031215612525575f80fd5b8135600681106124b1575f80fd5b5f60208284031215612543575f80fd5b81357fffffffff00000000000000000000000000000000000000000000000000000000811681146124b1575f80fd5b7f4e487b71000000000000000000000000000000000000000000000000000000005f52601160045260245ffd5b8181038181111561198a5761198a612572565b80516dffffffffffffffffffffffffffff8116811461209a575f80fd5b5f805f606084860312156125e1575f80fd5b6125ea846125b2565b92506125f8602085016125b2565b9150604084015163ffffffff81168114612610575f80fd5b809150509250925092565b5f7f8000000000000000000000000000000000000000000000000000000000000000820361264b5761264b612572565b505f0390565b5f5b8381101561266b578181015183820152602001612653565b50505f910152565b5f815180845261268a816020860160208601612651565b601f017fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffe0169290920160200192915050565b602081525f8251604060208401526126d76060840182612673565b905073ffffffffffffffffffffffffffffffffffffffff60208501511660408401528091505092915050565b5f73ffffffffffffffffffffffffffffffffffffffff8088168352861515602084015285604084015280851660608401525060a06080830152610b0960a0830184612673565b5f806040838503121561275a575f80fd5b505080516020909101519092909150565b60ff818116838216019081111561198a5761198a612572565b7f4e487b71000000000000000000000000000000000000000000000000000000005f52603260045260245ffd5b5f82516127c2818460208701612651565b9190910192915050565b5f602082840312156127dc575f80fd5b81516124b181612174565b84815283602082015273ffffffffffffffffffffffffffffffffffffffff83166040820152608060608201525f610d396080830184612673565b60e08152845160e08201525f602086015160028110612867577f4e487b71000000000000000000000000000000000000000000000000000000005f52602160045260245ffd5b610100830152604086015173ffffffffffffffffffffffffffffffffffffffff1661012083015260608601516128b661014084018273ffffffffffffffffffffffffffffffffffffffff169052565b50608086015161016083015260a086015160c06101808401526128dd6101a0840182612673565b91505061292c602083018673ffffffffffffffffffffffffffffffffffffffff808251168352602082015115156020840152806040830151166040840152506060810151151560608301525050565b8360a083015261294460c083018463ffffffff169052565b95945050505050565b8051602080830151919081101561298c577fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff8160200360031b1b821691505b50919050565b8082018082111561198a5761198a612572565b5f826129d8577f4e487b71000000000000000000000000000000000000000000000000000000005f52601260045260245ffd5b500490565b808202811582820484141761198a5761198a61257256fea2646970667358221220ef65a171ad62d36c692f403c0d72d7b67bc9465e0fecc0410f318eed9707357464736f6c63430008140033").unwrap(),
                ));

                db.insert_account_info(contract_address, acc_info.clone());

                stopwatch.stop();
                println!("total time inject contract and add liquidity: {:?}", stopwatch.elapsed());

                let mut calls = calls.into_iter().peekable();

                stopwatch.restart();

                let a= Bytes::from_hex("0xf46420d5").unwrap().to_vec();
                let c= Bytes::from_hex("000000000000000000000000000000000000000000000000003a43eaf2507ed8").unwrap().to_vec();
                let b=  Bytes::from_hex("00000000000000000000000000000000000000000000000000000000000000400000000000000000000000000000000000000000000000000000000000000003000000000000000000000000000000000000000000000000000000000000000200000000000000000000000088e6a0c2ddd26feeb64f039a2c41296fcb3f56400000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000c02aaa39b223fe8d0a0e5c4f27ead9083c756cc2000000000000000000000000a0b86991c6218b36c1d19d4a2e9eb0ce3606eb4800000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000020000000000000000000000002bc62cf3d2edc11557e862a324dc7c343e6ca7bc0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000a0b86991c6218b36c1d19d4a2e9eb0ce3606eb48000000000000000000000000176c9c91c16bd7dfdecc578b5205edbd071a617a0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000002000000000000000000000000b5d3980bbc8cbf8f00cc3d5faf10dfeade0714510000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000176c9c91c16bd7dfdecc578b5205edbd071a617a000000000000000000000000c02aaa39b223fe8d0a0e5c4f27ead9083c756cc2000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000").unwrap().to_vec();
                let mut old = alloy_primitives::ruint::aliases::U256::ZERO;
                for n in 0..1000 {
                    old += alloy_primitives::ruint::aliases::U256::from(1000000);

                    let x = U256::from_be_slice(&c);
                    let z: [u8; 32] = (x + old).to_be_bytes();

                    let mut f = Vec::<u8>::new();

                    for c in a.iter() {
                        f.push(*c);
                    }

                    for z in z.iter() {
                        f.push(*z);
                    }

                    for b in b.iter() {
                        f.push(*b);
                    }

                    while let Some((mut call, trace_types)) = calls.next() {

                        let g = Bytes::from(f.clone());

                        let h = TransactionInput
                        {
                            input: Some(g.clone()),
                            data: Some(g)
                        };

                        call.input = h;

                        let env = this.eth_api().prepare_call_env(
                            cfg.clone(),
                            block_env.clone(),
                            call,
                            &mut db,
                            Default::default(),
                        )?;
                        let config = TracingInspectorConfig::from_parity_config(&trace_types);
                        let mut inspector = TracingInspector::new(config);
                        let (res, _) = this.eth_api().inspect(&mut db, env, &mut inspector)?;
    
                        let trace_res = inspector
                            .into_parity_builder()
                            .into_trace_results_with_state(&res, &trace_types, &db)
                            .map_err(Eth::Error::from_eth_err)?;
    
                        results.push(trace_res);
    
                        // need to apply the state changes of this call before executing the
                        // next call
                        if calls.peek().is_some() {
                            // need to apply the state changes of this call before executing
                            // the next call
                            db.commit(res.state)
                        }

                    }
                }

                stopwatch.stop();
                println!("total time run 1000 times: {:?}", stopwatch.elapsed());

                Ok(results)
            })
            .await
    }
}

#[async_trait]
impl<Provider, Eth> TraceApiServer for TraceApi<Provider, Eth>
where
    Provider: BlockReader<Block = <Eth::Provider as BlockReader>::Block>
        + StateProviderFactory
        + EvmEnvProvider
        + ChainSpecProvider<ChainSpec: EthereumHardforks>
        + 'static,
    Eth: TraceExt + 'static,
{
    /// Executes the given call and returns a number of possible traces for it.
    ///
    /// Handler for `trace_call`
    async fn trace_call(
        &self,
        call: TransactionRequest,
        trace_types: HashSet<TraceType>,
        block_id: Option<BlockId>,
        state_overrides: Option<StateOverride>,
        block_overrides: Option<Box<BlockOverrides>>,
    ) -> RpcResult<TraceResults> {
        let _permit = self.acquire_trace_permit().await;
        let request =
            TraceCallRequest { call, trace_types, block_id, state_overrides, block_overrides };
        Ok(Self::trace_call(self, request).await.map_err(Into::into)?)
    }

    /// Handler for `trace_callMany`
    async fn trace_call_many(
        &self,
        calls: Vec<(TransactionRequest, HashSet<TraceType>)>,
        block_id: Option<BlockId>,
    ) -> RpcResult<Vec<TraceResults>> {
        let _permit = self.acquire_trace_permit().await;
        Ok(Self::trace_call_many(self, calls, block_id).await.map_err(Into::into)?)
    }

    /// Handler for `trace_rawTransaction`
    async fn trace_raw_transaction(
        &self,
        data: Bytes,
        trace_types: HashSet<TraceType>,
        block_id: Option<BlockId>,
    ) -> RpcResult<TraceResults> {
        let _permit = self.acquire_trace_permit().await;
        Ok(Self::trace_raw_transaction(self, data, trace_types, block_id)
            .await
            .map_err(Into::into)?)
    }

    /// Handler for `trace_replayBlockTransactions`
    async fn replay_block_transactions(
        &self,
        block_id: BlockId,
        trace_types: HashSet<TraceType>,
    ) -> RpcResult<Option<Vec<TraceResultsWithTransactionHash>>> {
        let _permit = self.acquire_trace_permit().await;
        Ok(Self::replay_block_transactions(self, block_id, trace_types)
            .await
            .map_err(Into::into)?)
    }

    /// Handler for `trace_replayTransaction`
    async fn replay_transaction(
        &self,
        transaction: B256,
        trace_types: HashSet<TraceType>,
    ) -> RpcResult<TraceResults> {
        let _permit = self.acquire_trace_permit().await;
        Ok(Self::replay_transaction(self, transaction, trace_types).await.map_err(Into::into)?)
    }

    /// Handler for `trace_block`
    async fn trace_block(
        &self,
        block_id: BlockId,
    ) -> RpcResult<Option<Vec<LocalizedTransactionTrace>>> {
        let _permit = self.acquire_trace_permit().await;
        Ok(Self::trace_block(self, block_id).await.map_err(Into::into)?)
    }

    /// Handler for `trace_filter`
    ///
    /// This is similar to `eth_getLogs` but for traces.
    ///
    /// # Limitations
    /// This currently requires block filter fields, since reth does not have address indices yet.
    async fn trace_filter(&self, filter: TraceFilter) -> RpcResult<Vec<LocalizedTransactionTrace>> {
        Ok(Self::trace_filter(self, filter).await.map_err(Into::into)?)
    }

    /// Returns transaction trace at given index.
    /// Handler for `trace_get`
    async fn trace_get(
        &self,
        hash: B256,
        indices: Vec<Index>,
    ) -> RpcResult<Option<LocalizedTransactionTrace>> {
        let _permit = self.acquire_trace_permit().await;
        Ok(Self::trace_get(self, hash, indices.into_iter().map(Into::into).collect())
            .await
            .map_err(Into::into)?)
    }

    /// Handler for `trace_transaction`
    async fn trace_transaction(
        &self,
        hash: B256,
    ) -> RpcResult<Option<Vec<LocalizedTransactionTrace>>> {
        let _permit = self.acquire_trace_permit().await;
        Ok(Self::trace_transaction(self, hash).await.map_err(Into::into)?)
    }

    /// Handler for `trace_transactionOpcodeGas`
    async fn trace_transaction_opcode_gas(
        &self,
        tx_hash: B256,
    ) -> RpcResult<Option<TransactionOpcodeGas>> {
        let _permit = self.acquire_trace_permit().await;
        Ok(Self::trace_transaction_opcode_gas(self, tx_hash).await.map_err(Into::into)?)
    }

    /// Handler for `trace_blockOpcodeGas`
    async fn trace_block_opcode_gas(&self, block_id: BlockId) -> RpcResult<Option<BlockOpcodeGas>> {
        let _permit = self.acquire_trace_permit().await;
        Ok(Self::trace_block_opcode_gas(self, block_id).await.map_err(Into::into)?)
    }

    /// Handler for `trace_callManyCustom`
    async fn trace_call_many_custom(
        &self,
        calls: Vec<(TransactionRequest, HashSet<TraceType>)>,
        block_id: Option<BlockId>,
    ) -> RpcResult<Vec<TraceResults>> {
        let _permit = self.acquire_trace_permit().await;
        Ok(Self::trace_call_many_custom(self, calls, block_id).await.map_err(Into::into)?)
    }
}

impl<Provider, Eth> std::fmt::Debug for TraceApi<Provider, Eth> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TraceApi").finish_non_exhaustive()
    }
}
impl<Provider, Eth> Clone for TraceApi<Provider, Eth> {
    fn clone(&self) -> Self {
        Self { inner: Arc::clone(&self.inner) }
    }
}

struct TraceApiInner<Provider, Eth> {
    /// The provider that can interact with the chain.
    provider: Provider,
    /// Access to commonly used code of the `eth` namespace
    eth_api: Eth,
    // restrict the number of concurrent calls to `trace_*`
    blocking_task_guard: BlockingTaskGuard,
}

/// Helper to construct a [`LocalizedTransactionTrace`] that describes a reward to the block
/// beneficiary.
fn reward_trace(header: &Header, reward: RewardAction) -> LocalizedTransactionTrace {
    LocalizedTransactionTrace {
        block_hash: Some(header.hash_slow()),
        block_number: Some(header.number),
        transaction_hash: None,
        transaction_position: None,
        trace: TransactionTrace {
            trace_address: vec![],
            subtraces: 0,
            action: Action::Reward(reward),
            error: None,
            result: None,
        },
    }
}
