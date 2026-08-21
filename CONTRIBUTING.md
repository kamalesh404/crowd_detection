# Contributing to CrowdSafe AI

Thank you for your interest in contributing to CrowdSafe AI! This document provides guidelines for contributing to this crowd safety system.

## How to Contribute

### Reporting Bugs

1. Check existing [issues](https://github.com/kamalesh404/crowd_detection/issues) to avoid duplicates
2. Open a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - Python version and OS

### Suggesting Features

1. Open an issue with the `enhancement` label
2. Describe the feature, use case, and implementation idea

### Submitting Changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Test locally (see Development Setup)
5. Commit with a clear message
6. Push and open a Pull Request

## Development Setup

```bash
git clone https://github.com/your-username/crowd_detection.git
cd crowd_detection
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
pip install -r requirements.txt
cp .env.example .env
python main.py
```

## Code Style

- Follow PEP 8 for Python code
- Use type hints where appropriate
- Keep functions focused and documented
- Add docstrings for public functions

## Testing

- Test changes with different video sources (webcam, video file)
- Verify detection accuracy is maintained
- Check performance impact (FPS should remain stable)

## Pull Request Guidelines

- PR title should be descriptive
- Reference related issues
- Describe what changed and why
- Include screenshots/video if visual changes

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
