# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The transport-neutral agent loop.

`loop` drives one rollout, `policy` decides budgets and stop conditions, and
`prompts` builds the task text.

The tool surface the loop dispatches — schema, implementation and result
compaction — lives in `synclaw.tools`.
"""
