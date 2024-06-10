# VisionQuery

VisionQuery is a comprehensive project designed to process and understand multimodal inputs, such as images and text queries, leveraging state-of-the-art models from the Hugging Face ecosystem. The system processes input through multiple specialized models and synthesizes the results using LLMs to provide coherent and insightful outputs.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Models Used](#models-used)
- [Contributing](#contributing)

## Overview

VisionQuery is designed to handle various types of inputs and process them through different models to generate a final, meaningful output. The project utilizes the following Hugging Face models:

- BLIP (Bootstrapping Language-Image Pre-training)
- ViLT (Vision-and-Language Transformer)
- GIT (Graph Induction Transformer)
- Pix2Struct

## Features

- **Image Processing**: Extracts meaningful information and generates descriptive text from images.
- **Visual Question Answering**: Handles queries related to visual content.
- **Graph and Table Understanding**: Converts visual representations of structured data into usable formats.
- **Multimodal Output Synthesis**: Combines outputs from various models to produce a coherent final result using GPT-3.5.

## Architecture

The architecture of VisionQuery is designed to integrate multiple models for processing inputs and synthesizing outputs. The flow of data is as follows:

<img src="https://i.imgur.com/PVQbzOb.jpeg" width=50% height=50% >

1. **Input Layer**: Accepts images and text queries.
2. **Hugging Face Models**:
   - **BLIP**: Processes images.
   - **ViLT**: Handles visual and language tasks.
   - **GIT**: Processes graphs and text with and without conditions.
   - **Pix2Struct**: Converts visual data into structured formats.
3. **Internal Outputs**: Intermediate results from each model.
4. **Language Model**: LLMs synthesizes the internal outputs into a final result.
5. **Final Output**: The processed and coherent result presented to the user.

## Models Used
- **BLIP (Bootstrapping Language-Image Pre-training)**: Aligns images and text for descriptive text generation.
- **ViLT (Vision-and-Language Transformer)**: Handles tasks requiring visual and textual understanding.
- **GIT (Graph Induction Transformer)**: Processes graph and text data with or without specific conditions.
- **Pix2Struct**: Converts visual data like tables and charts into structured formats.

## Contributing
- We welcome contributions to VisionQuery! If you have suggestions for improvements or new features, please open an issue or submit a pull request.
