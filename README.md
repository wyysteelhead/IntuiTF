<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/wyysteelhead/GeneticDVR">
    <img src="images/teaser.png" alt="Logo" width="700" height="350">
  </a>

  <h1 align="center">What You Think Is What You Get</h3>

  <p align="center">
  We present WYTWYG (What You Think Is What You Get), a novel framework that bridges the semantic gap between user intent and transfer function design in Direct Volume Rendering. Our system allows users to intuitively specify visualization goals through multimodal interactions through two core innovations: (1) an evolution-based TF space explorer that effectively navigates the vast parameter space, and (2) a generalized quality evaluator powered by Multi-modal Large Language Models (MLLMs) that provides intelligent visual guidance. Through these components, users can intuitively express their visualization goals while the system automatically optimizes TFs to match their intent. Our framework demonstrates superior generalizability across various volumetric datasets and significantly improves the efficiency of TF design compared to traditional approaches.
  </p>
</div>

## 🎬 Demo

Below is a demonstration of the GeneticDVR system in action:

<div align="center">
  <video src="images/demo.mp4" width="800" controls>
    Your browser does not support the video tag.
  </video>
</div>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#Installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#Evaluation">Evaluation</a></li>
    <li><a href="#BibTeX">BibTeX</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->

## About The Project

GeneticDVR is an intuitive transfer function (TF) design system for volumetric data visualization. Our goal is to make the process of specifying visualization goals more natural and efficient for users.

Here's what makes GeneticDVR special:
* 🚀 **Intuitive Interaction:** Specify visualization goals using multimodal interactions.
* 🤖 **Automated TF Design:** Combines visual feedback to automatically refine transfer functions.
* 🧠 **Smart Evaluation:** Uses a vision-language model to efficiently evaluate visualization quality.
* 📊 **Generalizable & Efficient:** Works with various datasets and achieves desirable visual results faster than traditional methods.

Compared to existing approaches, GeneticDVR offers:
- More intuitive user interaction
- Improved automation
- Stronger generalizability
- Higher efficiency

We welcome suggestions and contributions! Feel free to fork the repo, create a pull request, or open an issue.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- 🚀 Getting Started -->
## 🚀 Getting Started

Follow these steps to set up GeneticDVR locally.

### 🛠️ Prerequisites

Make sure you have the following installed:
* Python 3.8.8
* CUDA (for GPU acceleration)
* Conda
* DiffDVR (https://github.com/shamanDevel/DiffDVR)

### 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/wyysteelhead/GeneticDVR.git
cd GeneticDVR
```

2. Create and activate the conda environment:
```bash
conda create -n geneticdvr python=3.8.8
conda activate geneticdvr
```

3. Install DiffDVR first:
```bash
git clone https://github.com/shamanDevel/DiffDVR.git
cd DiffDVR
# Follow DiffDVR installation instructions
cd ..
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. (Optional) Download sample volumetric datasets from the [datasets](https://github.com/wyysteelhead/GeneticDVR/tree/main/datasets) folder.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- 🧑‍💻 Usage -->
## 🧑‍💻 Usage

### Using Task Scripts

The project provides task scripts for different use cases:

```bash
# For image processing tasks
./scripts/tasks/task_image.sh

# For video processing tasks
./scripts/tasks/task_video.sh
```

### Direct Command Usage

You can also run the genetic optimization directly using the `genetic.py` script. Here's the complete command structure:

```bash
python genetic_optimize/genetic.py \
    --base_url YOUR_API_URL \
    --api_key YOUR_API_KEY \
    --config_file "/path/to/config.json" \
    --prompt_folder ./prompt \
    --population_size 50 \
    --generations 30 \
    --save_interval 1 \
    --instruct_number "task_name" \
    --save_path "/path/to/save/results" \
    --quality_metrics "16,11,14" \
    --text_metrics "5" \
    --text_interval "16" \
    --bg_color "(255,255,255)" \
    --style_image "/path/to/style/image.png" \
    --model_name "gemini-2.0-flash-001"
```

#### Command Parameters

- `--base_url`: API endpoint URL
- `--api_key`: Your API key
- `--config_file`: Path to the dataset configuration file, our example config file locates in /root/autodl-tmp/GeneticDVR/diffdvr/config-files
- `--prompt_folder`: Directory containing prompt templates (default: ./prompt)
- `--population_size`: Size of the genetic algorithm population
- `--generations`: Number of generations to run
- `--save_interval`: How often to save results (default: 1)
- `--instruct_number`: Unique identifier for this run
- `--save_path`: Where to save the results
- `--quality_metrics`: Quality assessment metrics defined in GeneticDVR/prompt/aspects.json (comma-separated, default: "16,11,14")
- `--text_metrics`: Text-based metrics (default: "5")
- `--text_interval`: Interval for text-based evaluation (default: "16")
- `--bg_color`: Background color in RGB format (default: "(255,255,255)")
- `--style_image`: Path to style reference image
- `--model_name`: Name of the model to use (default: "gemini-2.0-flash-001")

#### Example Configurations

1. Image-based transfer function design:
```bash
python genetic_optimize/genetic.py \
    --config_file "/root/autodl-tmp/GeneticDVR/diffdvr/config-files/feet.json" \
    --population_size 50 \
    --generations 30 \
    --save_path "/folderFromHost/results/feet_image" \
    --instruct_number "feet_basic" \
    --style_image "/path/to/style/image.png"
```

2. Text-based transfer function design:
```bash
python genetic_optimize/genetic.py \
    --config_file "/root/autodl-tmp/GeneticDVR/diffdvr/config-files/engine.json" \
    --population_size 100 \
    --generations 50 \
    --save_interval 5 \
    --save_path "./results/engine_text" \
    --instruct_number "engine1" \
    --quality_metrics "16,11,14" \
    --text_metrics "5" \
    --text_interval "16"
```

Note: The script is typically run in a tmux session for long-running tasks. You can use the task scripts which handle this automatically, or set up your own tmux session:

```bash
# Create a new tmux session
tmux new-session -d -s "task_name"

# Send the command to the tmux session
tmux send-keys -t "task_name" "conda activate genetic" C-m
tmux send-keys -t "task_name" "python genetic_optimize/genetic.py [parameters...]" C-m
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Project Structure

```
GeneticDVR/
├── diffdvr/              # DiffDVR integration and configuration
│   ├── config-files/     # Configuration files for different datasets
│   ├── parametrizations.py # Parameter definitions
│   ├── settings.py       # Global settings
│   └── utils.py         # Utility functions
├── genetic.py           # Main entry point for genetic algorithm
├── genetic_optimize/    # Core genetic optimization implementation
│   ├── api/            # LLM API integration
│   │   ├── llmapi.py   # Base LLM API interface
│   │   ├── openai.py   # OpenAI API implementation
│   │   └── gemini.py   # Google Gemini API implementation
│   ├── config/         # Configuration management
│   │   ├── config_manager.py # Configuration handling
│   │   └── default_config.json # Default configuration
│   ├── eval/           # Evaluation and ranking modules
│   │   ├── metric_eval.py # Baseline metric-based evaluation
│   │   ├── llm_eval.py # LLM-based evaluation
│   │   └── elo_rating.py # ELO rating system for ranking
│   ├── states/         # State management and configuration
│   │   ├── bound.py    # Boundary settings of rendering parameters
│   │   ├── evaluation_state.py # Evaluation result state
│   │   ├── gaussian.py # Gaussian state
│   │   └── genetic_config.py # Genetic algorithm parameter state
│   ├── utils/          # Utility functions
│   │   ├── file_utils.py # File operations
│   │   ├── image_utils.py # Image processing
│   │   └── thread.py   # Threading utilities
│   └── visualize/      # Visualization tools
│       └── gaussian_visualizer.py # Gaussian visualization
├── prompt/             # Prompt templates and assets
│   ├── instructions.json # Instructions examples
│   └── prompt_format.txt # Standard format of prompt
├── scripts/            # Task execution scripts
├── volumes/           # Volume data files, currently support cvol volume files
└── requirements.txt   # Project dependencies
```

## Component Relationships

1. **Core Components**:
   - `genetic.py`: The main implementation file that contains the genetic algorithm for transfer function optimization
   - `genetic_optimize/`: Contains the genetic optimization implementation
   - `diffdvr/`: Integration with DiffDVR for differentiable volume rendering

2. **Task Scripts**:
   - `scripts/tasks/`: Contains shell scripts for different use cases
   - Each script:
     - Sets up the environment
     - Runs the genetic optimization
     - Saves results

3. **Data Flow**:
   ```
   Volume Data → genetic.py → DiffDVR → Results
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- NEWS SECTION -->
<!-- ## 📰 News

Stay up to date with the latest developments and updates for this project.

- 🚀 **2024-06-01**: Initial release of the GeneticDVR project.
- 🖱️ **2024-06-10**: Added support for multimodal interaction in transfer function design.
- 🤖 **2024-06-15**: Integrated vision-language model for visualization quality evaluation. -->

## ✅ TODO

Here are some planned features and improvements for GeneticDVR:

- [ ] Add support for additional volumetric data formats
- [ ] Improve user interface for transfer function editing
- [ ] Enhance documentation with more usage examples
- [ ] Implement real-time collaborative editing
- [ ] Expand evaluation metrics for visualization quality


<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- 📄 License -->
## 📄 License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- 📬 Contact -->
## 📬 Contact

Project Lead: wyysteelhead  
Email: wyysteelhead@gmail.com

Project Link: [https://github.com/wyysteelhead/GeneticDVR](https://github.com/wyysteelhead/GeneticDVR)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## 📚 BibTeX

If you use GeneticDVR in your research, please cite our work:

```bibtex
@misc{geneticdvr2024,
  title        = {GeneticDVR: Intuitive Transfer Function Design for Volumetric Data Visualization},
  author       = {Your Name and Collaborators},
  year         = {2024},
  howpublished = {\url{https://github.com/wyysteelhead/GeneticDVR}},
  note         = {Initial release}
}
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>
