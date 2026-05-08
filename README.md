# ShadowVault

**Your persistent, intelligent second brain for Grok (and beyond).**

A clean, robust, single-file personal knowledge base with **scored retrieval**, smart deduplication, importance-based pruning, rediscovery mode, and first-class Grok conversation migration.

Built for researchers, writers, deep thinkers, and heavy Grok users who want their insights to actually survive across sessions.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-99%20passing-brightgreen)

## Features

- **Smart scored retrieval** — relevance + importance + recency + rarity + tag bonus
- **Rediscovery mode** — surfaces forgotten but valuable entries
- **Grok-native migration** — one-prompt → JSON → perfect import workflow
- **Intelligent deduplication** — Jaccard similarity with automatic reinforcement
- **Safe atomic persistence** — never lose data on crash, rotating backups
- **Automatic pruning** — low-value entries cleaned with safety caps
- **99 comprehensive tests** — battle-tested and production-ready
- **Zero dependencies** — pure Python, works anywhere
- **pip install ready** — `pyproject.toml` included for future PyPI release

## Quick Start

### Option 1: One-file (instant)
```bash
curl -O https://raw.githubusercontent.com/PLRTCore/shadow-vault/main/shadow_vault.py
python shadow_vault.py
```

Option 2: Install as proper CLI (recommended)
```bash
pip install shadow-vault
shadow-vault
```
Note: `pyproject.toml` is included in this repo. Once published to PyPI, Option 2 will become available.

Then type `Shadow Vault: Help` to see all commands.

## Grok Migration Workflow (the killer feature)

1. In ShadowVault type: `Shadow Vault: Prompt`
2. Copy the printed prompt and paste it into any Grok chat
3. Grok returns a clean JSON array
4. Save it as `grok_export.json`
5. In ShadowVault type: `Shadow Vault: Migrate` and select the file
6. Done — your entire conversation history is now in your vault with proper importance + tags

## Command Reference

See `Shadow Vault: Help` inside the tool.

## Why ShadowVault?

Most note-taking tools forget context. ShadowVault **remembers what matters** — intelligently.

Perfect for long research threads, building a true second brain, or never losing a key insight again.

## Tech Highlights

- Jaccard-based deduplication with recent-entry window
- Composite scoring engine (tuned weights)
- Atomic writes + backup rotation
- Legacy format migration
- Full test coverage (tokenizer, migration, pruning, persistence, etc.)

## Contributing

Contributions welcome! Open an issue or PR.

## License

MIT © 2026 PLRTCore
