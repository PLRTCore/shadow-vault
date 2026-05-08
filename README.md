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

## Quick Start

### Option 1: One-file (instant)
```bash
curl -O https://raw.githubusercontent.com/PLRTCore/shadow-vault/main/shadow_vault.py
python shadow_vault.py


