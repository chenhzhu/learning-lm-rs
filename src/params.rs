use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::SafeTensors;

pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    pub(crate) fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        // DONE ("实现从safetensors文件的模型参数加载");
        let get_tensor = |name: &str| {
            let tensor_data = safetensor.tensor(name).expect(&format!("Tensor {} not found", name));
            let shape: Vec<usize> = tensor_data.shape().into_iter().map(|s| *s as usize).collect();
            let data: Vec<f32> = tensor_data.data()
                                            .chunks_exact(4)
                                            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                                            .collect();
            Tensor::new(data, &shape)  
        };
        
        

        let layers = config.num_hidden_layers;
        LLamaParams {
            embedding_table: get_tensor("lm_head.weight"),
            rms_att_w: (0..layers).map(|i| get_tensor(&format!("model.layers.{}.input_layernorm.weight", i))).collect(),
            wq: (0..layers).map(|i| get_tensor(&format!("model.layers.{}.self_attn.q_proj.weight", i))).collect(),
            wk: (0..layers).map(|i| get_tensor(&format!("model.layers.{}.self_attn.k_proj.weight", i))).collect(),
            wv: (0..layers).map(|i| get_tensor(&format!("model.layers.{}.self_attn.v_proj.weight", i))).collect(),
            wo: (0..layers).map(|i| get_tensor(&format!("model.layers.{}.self_attn.o_proj.weight", i))).collect(),
            rms_ffn_w: (0..layers).map(|i| get_tensor(&format!("model.layers.{}.post_attention_layernorm.weight", i))).collect(),
            w_up: (0..layers).map(|i| get_tensor(&format!("model.layers.{}.mlp.up_proj.weight", i))).collect(),
            w_gate: (0..layers).map(|i| get_tensor(&format!("model.layers.{}.mlp.gate_proj.weight", i))).collect(),
            w_down: (0..layers).map(|i| get_tensor(&format!("model.layers.{}.mlp.down_proj.weight", i))).collect(),
            rms_out_w: get_tensor("model.norm.weight"),
            lm_head: get_tensor("lm_head.weight"),
        }
    }
}

// {
//     "bos_token_id": 1, # 起始符token id
//     "eos_token_id": 2, # 结束符token id
//     "hidden_size": 128, # 隐藏层大小，即各层输出的最后一维
//     "intermediate_size": 384, # Feed-Forward神经网络的中间层大小
//     "max_position_embeddings": 512, # 最大序列长度
//     "num_attention_heads": 8, # Self-Attention的Q头数
//     "num_hidden_layers": 2, # 隐藏层数
//     "num_key_value_heads": 4, # Self-Attention的K和V头数
//     "rms_norm_eps": 1e-6, # RMS Normalization的epsilon参数
//     "rope_theta": 10000.0, # RoPE的theta参数
//     "tie_word_embeddings": true, # 起始和结束embedding参数矩阵是否共享同一份数据
//     "torch_dtype": "float32", # 模型数据类型
//     "vocab_size": 2048 # 词表大小
//   }

// ---- model::test_check_safetensor stdout ----
// Tensor name: model.layers.1.self_attn.k_proj.weight
// Tensor name: model.layers.1.mlp.down_proj.weight
// Tensor name: model.layers.0.mlp.up_proj.weight
// Tensor name: model.layers.0.post_attention_layernorm.weight
// Tensor name: model.layers.0.self_attn.k_proj.weight
// Tensor name: model.layers.0.self_attn.q_proj.weight
// Tensor name: model.layers.1.input_layernorm.weight
// Tensor name: model.layers.0.mlp.gate_proj.weight
// Tensor name: model.layers.0.input_layernorm.weight
// Tensor name: model.layers.0.self_attn.v_proj.weight
// Tensor name: model.layers.1.self_attn.v_proj.weight
// Tensor name: model.layers.1.mlp.gate_proj.weight
// Tensor name: model.layers.1.self_attn.o_proj.weight
// Tensor name: model.layers.1.mlp.up_proj.weight
// Tensor name: model.layers.1.self_attn.q_proj.weight
// Tensor name: lm_head.weight
// Tensor name: model.layers.0.mlp.down_proj.weight
// Tensor name: model.norm.weight
// Tensor name: model.layers.0.self_attn.o_proj.weight
// Tensor name: model.layers.1.post_attention_layernorm.weight