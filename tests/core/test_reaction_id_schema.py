# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A database made before template_id became reaction_id fails with the one-line fix."""

import sqlite3

import pytest

from synclaw.core import db
from synclaw.rollout.prompts import validate_model_run_database


@pytest.fixture
def old_database(tmp_path):
    path = tmp_path / "old.sqlite"
    db.init_db(path)
    with sqlite3.connect(path) as conn:
        conn.execute("ALTER TABLE reaction_templates RENAME COLUMN reaction_id TO template_id")
    return path


@pytest.mark.parametrize("entry_point", [db.init_db, validate_model_run_database])
def test_an_old_database_names_the_fix(old_database, entry_point):
    with pytest.raises(RuntimeError, match="RENAME COLUMN template_id TO reaction_id"):
        entry_point(old_database)


def test_the_named_fix_makes_the_database_usable(old_database):
    with sqlite3.connect(old_database) as conn:
        conn.execute("ALTER TABLE reaction_templates RENAME COLUMN template_id TO reaction_id")

    db.init_db(old_database)
