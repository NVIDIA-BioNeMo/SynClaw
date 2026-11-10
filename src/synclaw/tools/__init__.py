# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The model-facing tool surface: schema, implementation, and result compaction.

`contracts` declares what the model may call, `synthesis` implements it, and
`compaction` shrinks results before they go back. Adding a tool means touching
the first two side by side.

`contracts` and `compaction` are chemistry-free and must stay importable without
RDKit; the empty `__init__` is what guarantees that, not the package boundary.
"""
