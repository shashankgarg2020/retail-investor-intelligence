-- Retail Investor Intelligence: schema. Idempotent, safe to re-run.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS companies (
    id        SERIAL PRIMARY KEY,
    ticker    TEXT NOT NULL UNIQUE,
    name      TEXT NOT NULL,
    exchange  TEXT,
    industry  TEXT,
    aliases   JSONB NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS ingestion_runs (
    id               SERIAL PRIMARY KEY,
    platform         TEXT NOT NULL,
    company_id       INT NOT NULL REFERENCES companies(id),
    status           TEXT NOT NULL,              -- running | success | partial | failed | skipped
    is_synthetic     BOOLEAN NOT NULL DEFAULT FALSE,
    queries          JSONB NOT NULL DEFAULT '[]',
    fetched          INT NOT NULL DEFAULT 0,     -- normalised records seen
    inserted         INT NOT NULL DEFAULT 0,     -- new rows
    duplicates       INT NOT NULL DEFAULT 0,     -- already stored (platform id match)
    requests_ok      INT NOT NULL DEFAULT 0,
    requests_failed  INT NOT NULL DEFAULT 0,
    error            TEXT,
    checkpoint       TIMESTAMPTZ,                -- newest created_at seen; next run starts near here
    started_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at     TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS posts (
    id                 BIGSERIAL PRIMARY KEY,
    company_id         INT NOT NULL REFERENCES companies(id),
    platform           TEXT NOT NULL,            -- reddit | youtube | bluesky
    external_id        TEXT NOT NULL,            -- platform's own identifier
    content_type       TEXT NOT NULL,            -- post | comment | video
    title              TEXT,
    text               TEXT NOT NULL,
    text_hash          TEXT NOT NULL,
    author_id          TEXT,
    parent_external_id TEXT,
    parent_title       TEXT,                     -- context for comments
    created_at         TIMESTAMPTZ NOT NULL,     -- published on the platform
    collected_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    url                TEXT NOT NULL,
    likes              INT,                      -- NULL means unavailable, not zero
    comments           INT,
    shares             INT,
    views              BIGINT,
    language           TEXT,
    is_synthetic       BOOLEAN NOT NULL DEFAULT FALSE,
    extra              JSONB NOT NULL DEFAULT '{}',
    ingestion_run_id   INT REFERENCES ingestion_runs(id),
    tsv                tsvector GENERATED ALWAYS AS (
                           to_tsvector('english', coalesce(title, '') || ' ' || text)
                       ) STORED,
    -- Idempotent ingestion. company_id is part of the key so one post that
    -- mentions two tracked companies is stored once per company.
    UNIQUE (company_id, platform, external_id)
);

CREATE INDEX IF NOT EXISTS idx_posts_company_created ON posts (company_id, created_at);
CREATE INDEX IF NOT EXISTS idx_posts_text_hash ON posts (company_id, text_hash);
CREATE INDEX IF NOT EXISTS idx_posts_tsv ON posts USING GIN (tsv);

CREATE TABLE IF NOT EXISTS post_analysis (
    post_id              BIGINT PRIMARY KEY REFERENCES posts(id) ON DELETE CASCADE,
    relevance_label      TEXT NOT NULL,          -- relevant | possibly_relevant | irrelevant
    relevance_score      REAL NOT NULL,
    relevance_method     TEXT NOT NULL,          -- e.g. rules-v1 or llm:<model>
    relevance_reason     TEXT,
    quality_flag         TEXT NOT NULL,          -- ok | low_quality | promotional | potentially_automated
    duplicate_of         BIGINT REFERENCES posts(id) ON DELETE SET NULL,
    sentiment            TEXT,                   -- positive | neutral | negative
    sentiment_confidence REAL,                   -- model score, not a calibrated probability
    sentiment_model      TEXT,
    topic                TEXT,
    topic_score          REAL,
    topic_method         TEXT,
    processed_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_analysis_sentiment ON post_analysis (sentiment);
CREATE INDEX IF NOT EXISTS idx_analysis_topic ON post_analysis (topic);

CREATE TABLE IF NOT EXISTS embeddings (
    post_id   BIGINT PRIMARY KEY REFERENCES posts(id) ON DELETE CASCADE,
    model     TEXT NOT NULL,
    embedding vector(384) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw
    ON embeddings USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS aspect_sentiment (
    id        BIGSERIAL PRIMARY KEY,
    post_id   BIGINT NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    aspect    TEXT NOT NULL,
    sentiment TEXT NOT NULL,
    model     TEXT NOT NULL,
    UNIQUE (post_id, aspect)
);

-- The analysis_posts view is defined in 002 (it reads columns added there).
