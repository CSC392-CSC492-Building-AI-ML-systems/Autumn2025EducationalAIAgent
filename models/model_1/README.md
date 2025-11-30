# Model 1 - Terminal Session Annotator

Model 1 is a streamed annotator that processes terminal session events from XML files and generates annotations consisting of depth transitions and action summaries. The model uses vLLM for optimized inference and supports various evaluation and ablation modes.

## Overview

Model 1 processes terminal events in a streaming fashion, where each event receives:
- **Depth**: An integer indicating goal transitions (-1 = start subtask, 0 = continue, +N = finish N levels)
- **Summary**: A concise action-oriented summary (≤50 words by default)

The model processes events incrementally, using neighbor context (previous annotations) to maintain continuity.

## Requirements

Install dependencies from the root requirements file:
```bash
pip install -r ../requirements.txt
```

Additional dependencies for evaluation and ablation:
- `sentence-transformers` - For embedding-based similarity metrics
- `rouge-score` - For ROUGE-L metrics
- `bert-score` - For BERTScore F1 metrics
- `vllm` - For optimized LLM inference
- `lxml` - For XML parsing

## Running Model 1 Inference

Model 1 provides two main scripts for running inference:

1. **`model_1.py`** - Main inference script with flexible CLI and multiple output formats
   - Best for: Production inference, saving predictions to files, batch processing
   - Output: Saves predictions to file (JSONL, JSON, or TXT format)

2. **`scripts/model1_annotator.py`** - Core annotator with built-in GT evaluation support
   - Best for: Quick inference with automatic GT evaluation, interactive use
   - Output: Prints predictions and optionally computes GT evaluation metrics

**Quick Comparison:**
- Use `model_1.py` when you need structured output files and multiple format options
- Use `scripts/model1_annotator.py` when you want immediate GT evaluation metrics

### Basic Usage with model_1.py

The main entry point is `model_1.py`, which takes an XML input file and produces annotations:

```bash
python model_1.py <input_xml> <output_path>
```

**Example:**
```bash
python model_1.py ../../data/model_1/inputs/renee_rec2_parsed.xml annotations.jsonl
```

### Command-Line Arguments

- `input_xml` (required): Path to XML file containing `<event>` nodes
- `output_path` (required): Output path for predictions
- `--output-format`: Output format - `jsonl`, `json`, or `txt` (default: inferred from extension)
- `--no-fewshots`: Disable few-shot examples in the prompt
- `--k-target`: Number of target events per flush (default: 1)
- `--n-neigh`: Number of neighbor events to include as context (default: 200)
- `--max-events`: Limit annotation to first N events (for quick tests)

### Output Formats

The output format is inferred from the file extension unless `--output-format` is specified:

- **`.jsonl`** (default): JSON Lines format, one object per line:
  ```json
  {"idx": 0, "depth": -1, "summary": "Start backup task"}
  {"idx": 1, "depth": 0, "summary": "Verify backup archive"}
  ```

- **`.json`**: JSON array of objects:
  ```json
  [
    {"idx": 0, "depth": -1, "summary": "Start backup task"},
    {"idx": 1, "depth": 0, "summary": "Verify backup archive"}
  ]
  ```

- **`.txt`**: Legacy format with alternating depth/summary lines:
  ```
  -1
  Start backup task

  0
  Verify backup archive
  ```

### Environment Variables

You can override model parameters via environment variables:

```bash
export MODEL_ID="openai/gpt-oss-20b"           # Model identifier (default)
export GPU_UTIL=0.9                              # GPU memory utilization (default: 0.9)
export MAX_MODEL_LEN=131072                      # Maximum model context length (default)
export DTYPE="bfloat16"                          # Data type (default)
export MAX_NEW_TOKENS=2500                       # Maximum tokens to generate (default)
export SUMMARY_WORD_LIMIT=50                     # Maximum words in summaries (default)
export VLLM_TP=1                                 # Tensor parallel size (default: 1)
```

**Example with custom settings:**
```bash
MODEL_ID="openai/gpt-oss-20b" MAX_NEW_TOKENS=3000 \
python model_1.py input.xml output.jsonl
```

### Alternative: Using model1_annotator.py

For inference with automatic ground truth evaluation, you can use `scripts/model1_annotator.py`:

```bash
cd scripts
python model1_annotator.py --xml-path path/to/input.xml --gt-path path/to/gt.txt
```

**Command-Line Arguments:**
- `--xml-path`: Path to XML file containing `<event>` nodes (optional, uses default if not provided)
- `--gt-path`: Path to ground truth `.txt` file for evaluation (optional)
- `--no-eval`: Skip ground truth evaluation even if GT path exists

**Key Features:**
- Automatic GT evaluation when GT path is provided
- Prints predictions to console (no file output)
- Computes embedding-based similarity metrics (cosine, ROUGE-L, cross-encoder, BERTScore)
- Backward compatible: uses default paths from script if no arguments provided

For more details, see the [Running Inference with Ground Truth Evaluation](#running-inference-with-ground-truth-evaluation) section.

## Ablation Testing

The ablation runner (`scripts/model1_ablation_runner.py`) allows you to systematically test different prompt configurations and measure their impact on annotation quality.

### Ablation Modes

Four ablation modes are supported:

1. **`ablate_big`**: Big-block ablation - tests removing major prompt components
   - Full prompt (baseline)
   - No fewshots
   - No rules
   - No think-first block
   - No system role

2. **`ablate_few`**: Few-shot ablation - leave-one-out testing of individual examples

3. **`ablate_rules`**: Rules ablation - leave-one-out testing of individual rules

4. **`ablate_think`**: Think-first ablation - leave-one-out testing of think-first instructions

### Running Ablations

#### Single XML File (Backward Compatible)

```bash
cd scripts
python model1_ablation_runner.py <mode> > results.json
```

**Example:**
```bash
# Big-block ablation
python model1_ablation_runner.py ablate_big > big_ablation.json

# Few-shot ablation
python model1_ablation_runner.py ablate_few > fewshot_ablation.json

# Rules ablation
python model1_ablation_runner.py ablate_rules > rules_ablation.json

# Think-first ablation
python model1_ablation_runner.py ablate_think > think_ablation.json
```

#### Directory of XML Files

To run ablations over multiple XML files:

```bash
python model1_ablation_runner.py <mode> <path/to/xml_dir> > results.json
```

**Example:**
```bash
python model1_ablation_runner.py ablate_big ../../data/model_1/inputs/ > multi_big_ablation.json
```

The script will:
- Find all `*.xml` files in the directory
- Automatically infer ground truth paths for each XML
- Run the ablation for each file
- Compute metrics per XML and overall aggregated metrics
- Generate a human-readable summary file: `<mode>_metrics_summary.txt`

### Ablation Output

The ablation runner outputs JSON with:
- Per-XML results with detailed metrics for each ablation variant
- Overall aggregated metrics (weighted by number of annotation pairs)
- Timing and token statistics
- Full prompt and model output logs for each flush

**Metrics computed:**
- `cosine_mean`: Bi-encoder cosine similarity (sentence-transformers)
- `rougeL_mean`: ROUGE-L F1 score
- `cross_mean`: Cross-encoder STS similarity
- `bertF1_mean`: BERTScore F1

### Ground Truth Path Inference

The ablation runner automatically infers ground truth paths using these heuristics:

1. If XML name ends with `_parsed.xml`, try replacing with `_training.txt` in the same directory
2. If path contains an `inputs` segment, mirror it to `outputs` and replace `_parsed.xml` with `_training.txt`
3. Fall back to `GT_PATH` if set in the module

**Example paths:**
- XML: `data/model_1/inputs/1727009556_parsed.xml`
- GT (inferred): `data/model_1/outputs/1727009556_training.txt`

## Running Inference with Ground Truth Evaluation

To run inference and automatically evaluate against ground truth, you can use either:
1. **`model1_annotator.py`** - Simple inference with optional GT evaluation (supports CLI arguments)
2. **`model1_ablation_runner.py`** - Ablation testing framework that includes GT evaluation

### Using model1_annotator.py (Recommended for Simple Evaluation)

The annotator script is the easiest way to run inference with GT evaluation:

```bash
cd scripts

# Override paths via command-line arguments
python model1_annotator.py \
    --xml-path path/to/input.xml \
    --gt-path path/to/gt.txt
```

The script will:
1. Load the XML file
2. Run inference on all events
3. Automatically compute embedding-based similarity metrics if GT path exists
4. Display metrics including cosine similarity, ROUGE-L, cross-encoder, and BERTScore

### Using the Ablation Runner

The ablation runner automatically evaluates against ground truth when available and can test multiple prompt configurations:

```bash
cd scripts

# For single XML file (uses XML_PATH/GT_PATH from model1_annotator or infers GT path)
python model1_ablation_runner.py ablate_big

# For explicit XML file (automatically infers GT path)
python model1_ablation_runner.py ablate_big path/to/input.xml
```

The ablation runner will:
1. Load the XML file
2. Infer or use the GT path
3. Run inference with different prompt configurations
4. Compute metrics comparing predictions to ground truth for each configuration
5. Output detailed metrics including cosine similarity, ROUGE-L, cross-encoder, and BERTScore

### Manual Ground Truth Evaluation

If you want to run inference separately and then evaluate, you can use `model1_annotator.py` which supports command-line arguments to override paths:

**Command-Line Arguments:**
- `--xml-path`: Path to XML file containing `<event>` nodes (overrides default in script)
- `--gt-path`: Path to ground truth `.txt` file for evaluation (overrides default in script)
- `--no-eval`: Skip ground truth evaluation even if GT path is provided

**Examples:**

```bash
cd scripts

# Use default paths from script
python model1_annotator.py

# Override XML path only
python model1_annotator.py --xml-path ../../data/model_1/inputs/renee_rec2_parsed.xml

# Override both XML and GT paths
python model1_annotator.py \
    --xml-path ../../data/model_1/inputs/renee_rec2_parsed.xml \
    --gt-path ../../data/model_1/outputs/renee_rec2_training.txt

# Run inference but skip evaluation
python model1_annotator.py --xml-path input.xml --no-eval
```

The script will automatically compute embedding-based similarity metrics if a GT path is provided and `--no-eval` is not specified.

**Alternative (Legacy Method):**
You can also set `XML_PATH` and `GT_PATH` directly in the script file and run without arguments:
```python
XML_PATH = "../../data/model_1/inputs/renee_rec2_parsed.xml"
GT_PATH = "../../data/model_1/outputs/renee_rec2_training.txt"
```

### Ground Truth Format

Ground truth files should be in the legacy `.txt` format with alternating depth/summary lines:

```
-1
Start backup task

0
Verify backup archive

1
Complete backup workflow

```

## Configuration

### Script Configuration

The main scripts have configurable paths at the top:

**`model_1.py`**: Uses command-line arguments (no hardcoded paths)

**`scripts/model1_annotator.py`**: 
- Has default paths at the top of the file (used when no CLI arguments provided)
- Supports command-line arguments `--xml-path` and `--gt-path` to override defaults
- Default paths:
  ```python
  XML_PATH = "../../data/model_1/inputs/renee_rec2_parsed.xml"
  GT_PATH = "../../data/model_1/outputs/renee_rec2_training.txt"
  ```

**`scripts/model1_ablation_runner.py`**: Uses `model1_annotator` module's XML_PATH and GT_PATH, or infers GT paths automatically

### Model Parameters

Key parameters that can be adjusted:

- `K_TARGET`: Number of target events per flush (default: 1)
- `N_NEIGH`: Number of neighbor events for context (default: 200)
- `INCLUDE_FEWSHOTS_DEFAULT`: Whether to include few-shot examples (default: True)
- `MAX_NEW_TOKENS`: Maximum tokens to generate (default: 2500)
- `SUMMARY_WORD_LIMIT`: Maximum words in summaries (default: 50)

## Examples

### Example 1: Basic Inference

```bash
# Run inference on a single XML file
python model_1.py \
    ../../data/model_1/inputs/1727009556_parsed.xml \
    outputs/1727009556_annotations.jsonl
```

### Example 2: Inference with Custom Settings

```bash
# Use custom model and disable fewshots
MODEL_ID="custom-model" python model_1.py \
    input.xml \
    output.jsonl \
    --no-fewshots \
    --k-target 2 \
    --n-neigh 100
```

### Example 3: Ablation on Single File

```bash
cd scripts
python model1_ablation_runner.py ablate_big > big_ablation_results.json

# Check the human-readable summary
cat ablate_big_metrics_summary.txt
```

### Example 4: Ablation on Multiple Files

```bash
cd scripts
python model1_ablation_runner.py ablate_few ../../data/model_1/inputs/ > few_ablation_all.json
```

### Example 5: Inference with GT Evaluation

```bash
cd scripts

# Method 1: Use command-line arguments (recommended)
python model1_annotator.py \
    --xml-path ../../data/model_1/inputs/renee_rec2_parsed.xml \
    --gt-path ../../data/model_1/outputs/renee_rec2_training.txt

# Method 2: Use default paths from script (edit paths in script file first)
python model1_annotator.py

# Method 3: Override only XML path (uses GT path from script or skips eval)
python model1_annotator.py --xml-path path/to/input.xml
```

## Output and Metrics

### Inference Output

The main inference script (`model_1.py`) outputs:
- Progress information for each flush
- Model output with thinking/reasoning
- Token usage statistics
- Final consolidated table of all predictions

### Ablation Output

The ablation runner outputs:
- **JSON**: Detailed results with all configurations, prompts, outputs, and metrics
- **Text summary**: Human-readable metrics summary file (`<mode>_metrics_summary.txt`)

### Evaluation Metrics

When ground truth is available, the following metrics are computed:

- **Cosine Similarity**: Semantic similarity using bi-encoder embeddings
- **ROUGE-L**: Overlap-based F1 score for summaries
- **Cross-Encoder STS**: Pairwise semantic similarity
- **BERTScore F1**: Contextual embedding-based F1 score

## Troubleshooting

### Common Issues

1. **No events loaded**: Ensure XML file contains valid `<event>` nodes
2. **GT path not found**: Check that ground truth file exists and path inference heuristics match your directory structure
3. **Model loading errors**: Verify `MODEL_ID` is correct and model is accessible
4. **Memory errors**: Reduce `GPU_UTIL` or `MAX_MODEL_LEN`, or use tensor parallelism with `VLLM_TP`

### Debugging Tips

- Use `--max-events` to limit processing for quick tests
- Check token usage output to understand prompt sizes
- Verify XML structure matches expected format
- Ensure GT file format matches expected alternating depth/summary structure

## File Structure

```
model_1/
├── model_1.py              # Main inference script (CLI-friendly)
├── scripts/
│   ├── model1_annotator.py       # Core annotator with GT evaluation
│   ├── model1_ablation_runner.py # Ablation testing framework
│   ├── model1_benchmark_runner.py # Performance benchmarking
│   └── ...                        # Other utility scripts
└── README.md               # This file
```

## Additional Resources

- See `Model_Evaluation_Workflow.md` in `data/llm_Evaluation/` for evaluation workflows
- Check individual script docstrings for detailed function documentation
- Refer to vLLM documentation for advanced model configuration

