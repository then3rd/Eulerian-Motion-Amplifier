# Video Magnification Project

A Python application for Eulerian video magnification with a graphical user interface built using PyQt6. \
This tool allows you to amplify subtle motions and color changes in videos that are normally invisible to the naked eye.

## Features

- Eulerian video magnification algorithm implementation
- Interactive GUI for easy video processing
- Support for various video formats

## Prerequisites

- Python 3.13
- `curl` (for downloading sample videos)
- `make` (optional, for using Makefile commands)

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd <project-directory>
```

### 2. Set up the environment and install dependencies

Using Make (recommended):

```bash
make local_packages
```

## Usage

### Quick Start

Run with the sample video:

```bash
make run
```

This will:
1. Download a sample video
2. Launch the application with GUI

## Development

### Install Git Pre-commit Hooks

```bash
make hooks
```

## License
```
The Happy Bunny License (Modified MIT License)
--------------------------------------------------------------------------------
Copyright (c) 2005 - 2014 G-Truc Creation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

Restrictions:
 By making use of the Software for military purposes, you choose to make a
 Bunny unhappy.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```
