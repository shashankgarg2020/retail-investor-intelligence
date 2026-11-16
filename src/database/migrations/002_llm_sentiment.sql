-- LLM investor-stance sentiment, post type and aspect sentiment. Idempotent, safe to re-run.
--
-- post_analysis.sentiment stays the label every metric uses. It now holds the LLM's
-- investor stance when one exists, otherwise FinBERT's label; sentiment_model says which.
-- FinBERT's own output is kept in finbert_* so the two can be evaluated side by side.

ALTER TABLE post_analysis ADD COLUMN IF NOT EXISTS finbert_sentiment   TEXT;
ALTER TABLE post_analysis ADD COLUMN IF NOT EXISTS finbert_confidence  REAL;
ALTER TABLE post_analysis ADD COLUMN IF NOT EXISTS post_type           TEXT;  -- opinion | news | market_data | question | other
ALTER TABLE post_analysis ADD COLUMN IF NOT EXISTS sentiment_reason    TEXT;

CREATE INDEX IF NOT EXISTS idx_analysis_post_type ON post_analysis (post_type);
CREATE INDEX IF NOT EXISTS idx_aspect_sentiment_aspect ON aspect_sentiment (aspect);

-- The single definition of "counts towards company analytics".
-- Relevance is separate from sentiment: irrelevant, low-quality, duplicate and
-- unsupported-language posts never reach sentiment or topic metrics.
DROP VIEW IF EXISTS analysis_posts;
CREATE VIEW analysis_posts AS
SELECT
    p.id, p.company_id, c.ticker, p.platform, p.external_id, p.content_type,
    p.title, p.text, p.author_id, p.parent_title, p.created_at, p.collected_at,
    p.url, p.likes, p.comments, p.shares, p.views, p.language, p.is_synthetic,
    a.relevance_label, a.relevance_score, a.relevance_method,
    a.sentiment, a.sentiment_confidence, a.sentiment_model, a.sentiment_reason,
    a.finbert_sentiment, a.post_type,
    a.topic, a.topic_score,
    CASE a.sentiment WHEN 'positive' THEN 1 WHEN 'negative' THEN -1 ELSE 0 END AS sentiment_value,
    ln(1 + coalesce(p.likes, 0) + coalesce(p.comments, 0) + coalesce(p.shares, 0)) AS engagement_weight
FROM posts p
JOIN companies c ON c.id = p.company_id
JOIN post_analysis a ON a.post_id = p.id
WHERE a.relevance_label = 'relevant'
  AND a.quality_flag = 'ok'
  AND a.duplicate_of IS NULL
  AND p.language = 'en'
  AND a.sentiment IS NOT NULL;
