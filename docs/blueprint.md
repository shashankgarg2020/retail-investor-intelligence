# Curation AI Engineer Interview Task --- Retail Investor Intelligence

## 1. Executive summary

Build an MVP Retail Investor Intelligence Platform that helps Investor
Relations (IR) teams understand public conversations about three listed
companies---VEON, STB and TTWO---across three social platforms.

The product is not merely a sentiment dashboard. It should help an IR
professional understand:

-   What retail investors are discussing.
-   Why they are discussing it.
-   Which concerns are increasing or declining.
-   How discussions differ across platforms.
-   Which original posts support each insight.
-   What the collected evidence can and cannot establish.

The recommended MVP is a modular Python application with three ingestion
adapters, a canonical data model, a processing and analytics pipeline, a
PostgreSQL database with pgvector, a grounded AI assistant, and a simple
dashboard.

The key principle is to build one complete, explainable vertical slice
rather than many unfinished features.

------------------------------------------------------------------------

## 2. Understand the business problem

### 2.1 User and job to be done

The primary user is an Investor Relations professional at a listed
company. They need to monitor public conversations without manually
visiting several platforms and reading large volumes of posts.

A typical scenario:

1.  A company releases quarterly results.
2.  Retail investors discuss revenue, debt, growth, products, management
    and valuation.
3.  Different platforms develop different narratives.
4.  The IR team needs to identify the main themes and understand what is
    driving them.
5.  The team needs evidence to investigate or prepare communications.

The platform should provide a structured view of public discussion, not
make investment recommendations or predict share prices.

### 2.2 Translate the brief into requirements

  -----------------------------------------------------------------------
  Business question                   Product capability
  ----------------------------------- -----------------------------------
  What are investors discussing?      Topic extraction and ranking

  Is discussion positive or negative? Sentiment distribution and trends

  What is driving negative sentiment? Aspect sentiment and
                                      evidence-backed themes

  What changed this week?             Period-over-period trend detection

  Which platform has the most         Platform comparison
  discussion?                         

  What are people saying about a      Filtered post explorer and RAG
  particular issue?                   

  What should the IR team             Emerging concerns with supporting
  investigate?                        posts
  -----------------------------------------------------------------------

### 2.3 MVP screens

1.  **Company overview:** company selector, date range, post volume,
    sentiment, trends and platform breakdown.
2.  **Topic intelligence:** topic volume, sentiment, growth and
    representative posts.
3.  **Content explorer:** searchable posts with platform, date,
    engagement, relevance, sentiment, topic and source link.
4.  **AI assistant:** questions answered using collected content, with
    evidence citations and limitations.

All dashboard metrics must be computed from stored data. Any
illustrative or synthetic data must be clearly labelled.

------------------------------------------------------------------------

## 3. Select the social platforms

A platform choice should consider:

-   Relevance to retail-investor discussions.
-   Public content availability.
-   API access and terms.
-   Rate limits and cost.
-   Engagement metadata.
-   Ease of implementing a reliable adapter.
-   Data retention and deletion requirements.

### 3.1 Recommended MVP combination

  -----------------------------------------------------------------------
  Platform                Why it is useful        Integration
                                                  considerations
  ----------------------- ----------------------- -----------------------
  Reddit                  Long-form investor      Use an authorised Data
                          discussions, questions  API application; follow
                          and arguments           API terms and deletion
                                                  requirements

  YouTube                 Financial commentary,   Use YouTube Data API;
                          earnings analysis and   search and comment
                          video comments          retrieval have quota
                                                  implications

  Bluesky                 Short-form public       Public read APIs can
                          discussion and          make an MVP integration
                          reactions               relatively accessible
  -----------------------------------------------------------------------

This is a pragmatic implementation combination, not a claim that these
are necessarily the three largest retail-investor platforms.

Alternative combinations include Reddit + YouTube + X, or Reddit + X +
Stocktwits. X may introduce API cost and access dependencies. Stocktwits
API access may be restricted. Verify current access, commercial
permissions and terms before implementation.

If an integration cannot be authorised within the task timeframe,
implement its adapter interface and clearly label any fixture data as
synthetic. Never present synthetic records as live collected content.

### 3.2 Company configuration and entity resolution

Create a configuration for each ticker, including:

-   Ticker.
-   Legal/company name.
-   Exchange.
-   Aliases.
-   Subsidiaries and relevant products.
-   Industry-specific vocabulary.

Example configuration:

``` python
COMPANIES = {
    "VEON": {
        "name": "VEON Ltd.",
        "exchange": "NASDAQ",
        "aliases": ["VEON", "VEON Ltd", "VEON Group"],
        "industry": "Telecommunications",
    },
    "STB": {
        "name": "Sacombank",
        "exchange": "HOSE",
        "aliases": ["STB", "Sacombank", "Saigon Thuong Tin"],
        "industry": "Banking",
    },
    "TTWO": {
        "name": "Take-Two Interactive",
        "exchange": "NASDAQ",
        "aliases": [
            "TTWO",
            "Take-Two",
            "Take Two Interactive",
            "Rockstar Games",
        ],
        "industry": "Video Games",
    },
}
```

**Important:** confirm the intended company and exchange for STB with
the task owner. The example assumes Sacombank, but ticker symbols can be
ambiguous.

Do not assume every ticker or product-name mention is relevant. For
example, a post about playing GTA is not necessarily an investor
discussion about Take-Two.

Use two stages:

1.  **High-recall candidate retrieval:** ticker, company name, aliases,
    subsidiaries and products.
2.  **Relevance validation:** deterministic rules for obvious cases and
    a classifier or LLM for ambiguous cases.

Keep relevance separate from sentiment. A negative post that is
unrelated to the company must not enter company sentiment analytics.

------------------------------------------------------------------------

## 4. Proposed system architecture

``` text
Reddit API       YouTube API       Bluesky API
     |                |                 |
     +----------------+-----------------+
                      |
              Ingestion adapters
       pagination / retries / rate limits
                      |
                Raw records
                      |
          Normalisation and validation
                      |
       deduplication / entity resolution
        language / spam / quality checks
                      |
             Intelligence pipeline
       sentiment / topics / aspects
             embeddings / trends
                      |
             PostgreSQL + pgvector
                      |
             +--------+--------+
             |                 |
       Analytics API       RAG service
       SQL aggregations    hybrid retrieval
             |                 |
             +--------+--------+
                      |
               Streamlit UI
     overview / topics / explorer / AI chat
```

### 4.1 Recommended technology stack

  -----------------------------------------------------------------------
  Component               Choice                  Reason
  ----------------------- ----------------------- -----------------------
  Language                Python                  Strong data and ML
                                                  ecosystem

  Backend                 FastAPI                 Typed APIs, validation
                                                  and async support

  Frontend                Streamlit               Fast MVP UI development

  Database                PostgreSQL              Relational analytics
                                                  and reliable
                                                  persistence

  Vector search           pgvector                Semantic retrieval
                                                  without another
                                                  database initially

  Data processing         Pandas                  Convenient
                                                  transformations and
                                                  aggregations

  Sentiment               Finance-oriented        Better starting point
                          transformer baseline    for financial language

  Embeddings              Sentence Transformers   Local, cost-controlled
                                                  embeddings

  LLM                     OpenAI API or another   Grounded synthesis and
                          provider                structured outputs

  Scheduling              APScheduler initially   Simple periodic
                                                  ingestion

  Deployment              Docker Compose          Reproducible local
                                                  environment

  Testing                 Pytest                  Unit and integration
                                                  tests
  -----------------------------------------------------------------------

### 4.2 Why not use LangChain for everything?

A framework is not automatically required because an application
contains an LLM.

For the MVP, implement the core RAG steps explicitly:

-   Generate embeddings.
-   Query PostgreSQL/pgvector.
-   Apply metadata filters.
-   Build the prompt.
-   Call the LLM through a provider abstraction.
-   Validate the result with Pydantic.

This provides control and makes retrieval and prompt behaviour easier to
debug.

LangChain or LangGraph can be introduced when the workflow genuinely
needs more complex orchestration, tool routing, state management or
multi-step execution. Avoid adding a framework without a concrete
requirement.

### 4.3 Why PostgreSQL and pgvector?

The product needs both structured and unstructured retrieval:

-   Structured: company, platform, date, sentiment, topic, engagement.
-   Unstructured: the meaning of a post.

PostgreSQL supports filtering, joins, aggregation and full-text search.
pgvector adds vector similarity search. One database reduces deployment
and synchronisation complexity for an MVP.

Consider Qdrant, Pinecone or Weaviate later if independent vector
scaling, specialised features or workload characteristics justify them.

------------------------------------------------------------------------

## 5. Repository structure

``` text
retail-investor-intelligence/
│
├── README.md
├── .env.example
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
│
├── config/
│   ├── settings.py
│   ├── companies.yaml
│   └── platform_config.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── sample/
│
├── src/
│   ├── ingestion/
│   │   ├── base.py
│   │   ├── reddit.py
│   │   ├── youtube.py
│   │   ├── bluesky.py
│   │   └── orchestrator.py
│   │
│   ├── processing/
│   │   ├── normalizer.py
│   │   ├── deduplication.py
│   │   ├── entity_resolution.py
│   │   ├── language_detection.py
│   │   ├── spam_detection.py
│   │   └── engagement.py
│   │
│   ├── intelligence/
│   │   ├── sentiment.py
│   │   ├── aspect_sentiment.py
│   │   ├── topic_extraction.py
│   │   ├── trend_detection.py
│   │   ├── summarization.py
│   │   └── embeddings.py
│   │
│   ├── database/
│   │   ├── connection.py
│   │   ├── models.py
│   │   ├── repositories.py
│   │   └── migrations/
│   │
│   ├── rag/
│   │   ├── chunking.py
│   │   ├── retriever.py
│   │   ├── query_engine.py
│   │   ├── prompts.py
│   │   ├── reranker.py
│   │   └── response_schema.py
│   │
│   ├── analytics/
│   │   ├── metrics.py
│   │   ├── aggregations.py
│   │   └── insights.py
│   │
│   ├── api/
│   │   ├── main.py
│   │   ├── routes/
│   │   │   ├── companies.py
│   │   │   ├── posts.py
│   │   │   ├── analytics.py
│   │   │   └── chat.py
│   │   └── schemas/
│   │
│   └── frontend/
│       ├── app.py
│       ├── pages/
│       │   ├── overview.py
│       │   ├── topics.py
│       │   ├── explorer.py
│       │   └── assistant.py
│       └── components/
│
├── scripts/
│   ├── collect_data.py
│   ├── seed_database.py
│   ├── run_pipeline.py
│   └── evaluate.py
│
└── tests/
    ├── test_ingestion.py
    ├── test_normalization.py
    ├── test_entity_resolution.py
    ├── test_sentiment.py
    ├── test_retrieval.py
    └── test_api.py
```

This is the logical target structure. During a short MVP build, fewer
files are acceptable, provided the boundaries between ingestion,
processing, analytics, retrieval and UI remain clear.

------------------------------------------------------------------------

## 6. Data ingestion

### 6.1 Canonical data model

Each platform has a different response schema. Convert all source
records into one canonical model.

  Field                  Meaning
  ---------------------- -----------------------------------
  id                     Internal primary key
  external_id            Original platform identifier
  platform               reddit / youtube / bluesky
  company_ticker         VEON / STB / TTWO
  content_type           post / comment / video metadata
  text                   Normalised text
  author_id              Platform identifier, if permitted
  created_at             Original publication time
  collected_at           Time the application collected it
  url                    Original source URL
  likes                  Likes, nullable
  comments               Comment count, nullable
  shares                 Shares/reposts, nullable
  views                  Views, nullable
  language               Detected language
  relevance_score        Company relevance
  sentiment              positive / neutral / negative
  sentiment_confidence   Model confidence
  topic                  Assigned topic
  embedding              Vector representation

Important decisions:

-   Keep the platform's external ID to make ingestion idempotent.
-   Keep `created_at` and `collected_at` separate.
-   Treat unavailable engagement metrics as null, not zero.
-   Preserve source URLs.
-   Store raw payloads only where platform terms and retention policies
    permit.
-   Implement source-specific deletion and retention policies.

### 6.2 Adapter pattern

Each platform adapter should implement a common interface, for example:

-   `fetch_posts()`
-   `normalise()`
-   `get_next_page()`
-   `health_check()`

The orchestrator should not need to know platform-specific response
details.

### 6.3 Query strategy

Build company-specific search terms and aliases.

Examples:

-   VEON: `VEON`, `VEON Ltd`, `VEON earnings`, `VEON revenue`,
    `VEON debt`.
-   STB: `STB`, `Sacombank`, `Sacombank earnings`,
    `Sacombank financial results`.
-   TTWO: `TTWO`, `Take-Two Interactive`, `Take-Two earnings`, `GTA VI`,
    `Rockstar Games`.

Product terms should be validated for company relevance. Do not classify
every mention of a product as an investor discussion.

### 6.4 Pagination and incremental ingestion

For cursor-based APIs:

1.  Fetch a page.
2.  Normalise and persist its records.
3.  Read the next cursor.
4.  Fetch the next page.
5.  Stop at a configured page limit or date boundary.

For incremental collection:

-   First run: collect a defined historical window.
-   Later runs: collect records newer than the last successful
    checkpoint.
-   Use a small overlap window to catch late-arriving records.
-   Deduplicate using platform + external ID.
-   Store ingestion checkpoints per platform and query.

### 6.5 Reliability

Implement:

-   Request timeouts.
-   Retries with exponential backoff.
-   Maximum retry counts.
-   Rate-limit handling.
-   Structured error logs.
-   Per-platform run status.
-   Partial success.

If one source fails, the other sources should still be processed. The
dashboard should display source freshness and failures rather than
silently implying full coverage.

------------------------------------------------------------------------

## 7. Processing and data quality

Recommended pipeline:

1.  Raw ingestion.
2.  Text and timestamp normalisation.
3.  Deduplication.
4.  Language detection.
5.  Spam and quality filtering.
6.  Company entity resolution.
7.  Sentiment and topic enrichment.
8.  Embedding generation.
9.  Persistence.

### 7.1 Deduplication

Use:

-   Platform ID uniqueness.
-   Normalised text hashes.
-   Optional fuzzy matching for near-identical content.

Do not aggressively merge semantically similar posts. Two users
independently discussing the same concern are two separate opinions, not
duplicates.

### 7.2 Spam and low-quality content

MVP heuristics can include:

-   Repeated identical messages.
-   Excessive promotional links.
-   Repetitive low-information content.
-   Unrelated hashtags.
-   Suspiciously high posting frequency.

Avoid declaring an account to be a bot without reliable evidence. Use
cautious labels such as "potentially automated" or "low-quality
content".

### 7.3 Entity resolution

Possible outputs:

-   Relevant.
-   Possibly relevant.
-   Irrelevant.

Use deterministic matching for obvious cases and contextual
classification for ambiguous cases. Keep the score and the rule/model
version so the decision can be audited.

------------------------------------------------------------------------

## 8. Machine learning and analytics

### 8.1 Sentiment analysis

Start with three labels:

-   Positive.
-   Negative.
-   Neutral.

Financial language is different from general language. For example:

> "The stock dropped 15%, but this is an excellent buying opportunity."

A general sentiment model may overemphasise "dropped 15%" and miss the
positive investor stance.

Possible approaches:

  -----------------------------------------------------------------------
  Method                  Advantages              Limitations
  ----------------------- ----------------------- -----------------------
  VADER                   Lightweight and fast    Weak financial context

  FinBERT                 Finance-oriented        May miss retail slang
                          language                and sarcasm

  General transformer     Broad language          Domain mismatch
                          understanding           

  LLM classification      Flexible contextual     Cost, latency and
                          reasoning               output variability
  -----------------------------------------------------------------------

Recommended MVP: use a finance-oriented pretrained classifier as the
baseline, then compare it against a small manually labelled set. Use an
LLM for selected ambiguous cases or evaluation, not necessarily every
post.

Store:

-   Label.
-   Confidence.
-   Model name and version.
-   Processing timestamp.

Do not call confidence a calibrated probability unless calibration has
been evaluated.

### 8.2 Aspect-based sentiment

Overall sentiment can hide important differences.

Example:

"The company's revenue growth is impressive, but its debt levels are
worrying."

  Aspect           Sentiment
  ---------------- -----------
  Revenue growth   Positive
  Debt             Negative

Possible configurable aspects:

-   VEON: revenue, debt, network expansion, subscribers, regulation,
    management.
-   STB: asset quality, non-performing loans, deposits, net interest
    margin, credit growth, regulation.
-   TTWO: GTA VI, game releases, bookings, development costs,
    management, competition.

For the MVP, use structured LLM extraction on a limited dataset and
validate the output. Later, train or fine-tune a multi-label aspect
classifier if enough labelled examples exist.

### 8.3 Topic extraction

Two main approaches:

**Predefined topic classification**

Advantages: - Consistent categories. - Easy to explain. - Fast to
implement.

Disadvantages: - May miss unexpected themes. - Taxonomy needs
maintenance.

**BERTopic-style discovery**

Typical process: 1. Generate embeddings. 2. Reduce dimensionality with
UMAP. 3. Cluster with HDBSCAN. 4. Extract representative terms. 5.
Assign readable labels.

Advantages: - Can discover unexpected discussion clusters.

Disadvantages: - Sensitive to data volume and parameters. - Can create
fragmented or incoherent clusters. - Requires human validation.

Recommended MVP: predefined topic categories for stable reporting. Treat
unsupervised topic discovery as a later enhancement or a separately
evaluated experiment.

### 8.4 Trend detection

A topic is not necessarily emerging just because it is popular.

Compare current discussion volume with a previous equivalent period.

\[ Growth = `\frac{Current - Previous}{\max(Previous,\epsilon)}`{=tex}
`\times 100`{=tex} \]

When previous volume is zero, report absolute growth rather than an
infinite percentage.

Use minimum-volume thresholds to avoid overreacting to tiny counts.
Later, consider rolling baselines, EWMA or change-point detection.

### 8.5 Engagement-weighted sentiment

Calculate both:

-   **Post-weighted sentiment:** every unique post has equal weight.
-   **Engagement-weighted sentiment:** posts receive a weight based on
    observed engagement.

One possible weight:

\[ w_i = `\log`{=tex}(1 + Likes_i + Comments_i + Shares_i) \]

A sentiment score can be:

\[ S = `\frac{\sum_i w_i s_i}{\sum_i w_i}`{=tex} \]

where positive = +1, neutral = 0 and negative = -1.

Use post-weighted sentiment as the default. Engagement is not proof of
agreement and differs across platforms. Display the weighted measure
separately and label it clearly.

------------------------------------------------------------------------

## 9. RAG architecture

The AI assistant should answer questions about the collected dataset,
not rely on the LLM's general knowledge.

Example questions:

-   What are investors saying about VEON's financial performance?
-   What are the main concerns about TTWO?
-   Compare positive and negative discussions about STB.
-   What changed in the last seven days?
-   Which platform is driving the negative discussion?
-   Show evidence for the biggest investor concern.

### 9.1 RAG lifecycle

``` text
User question
     |
Query understanding
(company / dates / platform / intent)
     |
Metadata filtering
     |
Hybrid retrieval
(keyword + vector search)
     |
Reranking and evidence selection
     |
Context construction
(posts + metrics + URLs + timestamps)
     |
LLM generation
     |
Structured response validation
     |
Answer with evidence citations
```

### 9.2 Chunking

Social posts are already short. Treat each post or comment as an
independent retrieval document rather than splitting it into arbitrary
chunks.

For longer video descriptions or transcripts, use semantic or recursive
chunking. Preserve relationships to the parent video and source.

Attach metadata to every document:

-   Company ticker.
-   Platform.
-   Original post ID.
-   Content type.
-   Publication timestamp.
-   Sentiment.
-   Topic.
-   Relevance score.
-   Engagement.
-   Source URL.

### 9.3 Embeddings

Possible models include:

-   `all-MiniLM-L6-v2`: lightweight and fast.
-   BGE small variants: efficient retrieval embeddings.
-   Larger BGE variants: potentially stronger but more
    resource-intensive.
-   Managed embedding APIs: convenient but introduce API cost and
    external dependency.

Choose based on a small retrieval evaluation, not popularity alone.

### 9.4 Hybrid retrieval

Vector search finds semantically related content. Keyword search finds
exact terms and phrases. Metadata filters enforce company, platform,
sentiment and date constraints.

Use PostgreSQL full-text search plus pgvector similarity.

A ranking method such as Reciprocal Rank Fusion can combine ranked
lists:

\[ RRF(d) = `\sum`{=tex}\_{r
`\in `{=tex}R}`\frac{1}{k + rank_r(d)}`{=tex} \]

This is useful when one retriever finds exact ticker mentions and
another finds semantically related discussions.

### 9.5 SQL versus RAG

Do not use RAG for every question.

  Question                                    Best method
  ------------------------------------------- -----------------------
  How many posts were collected?              SQL
  What percentage are negative?               SQL
  What are investors concerned about?         RAG
  What are the main reasons for negativity?   SQL + RAG
  Compare Reddit and YouTube sentiment        SQL
  Summarise investor discussions              RAG
  What topics are emerging?                   Trend analytics + RAG

Numerical answers must come from deterministic queries. The LLM can
explain the results, but should not estimate counts from a handful of
retrieved documents.

### 9.6 Query planning

Classify questions into intents such as:

-   COUNT
-   SENTIMENT
-   TREND
-   TOPIC
-   COMPARISON
-   EXPLANATION
-   SUMMARY
-   EVIDENCE

Extract company, date range, platform, sentiment and result limit.
Initially use deterministic routing rather than an autonomous agent.

Example:

"What are the top three negative concerns about TTWO?"

1.  Identify ticker TTWO.
2.  Identify negative sentiment.
3.  Identify topic-ranking and explanation intent.
4.  Query SQL for the top negative themes.
5.  Retrieve representative posts for each theme.
6.  Ask the LLM to explain the findings using that evidence.
7.  Attach source citations.

### 9.7 Grounding and prompt injection

Treat social posts as untrusted data, never as instructions.

The prompt should require the model to:

1.  Use only supplied evidence for claims about collected discussion.
2.  Avoid inventing financial results or company facts.
3.  Distinguish investor opinions from verified facts.
4.  State when evidence is insufficient.
5.  Cite supporting post IDs.
6.  Ignore instructions embedded in retrieved content.
7.  Include the date range and platforms represented.

Validate that cited IDs exist in the retrieved context. If there is
insufficient evidence, abstain instead of inventing an answer.

### 9.8 Structured response

Return a validated schema containing:

-   Summary.
-   Main themes.
-   Supporting evidence IDs.
-   Platform breakdown.
-   Time period.
-   Number of posts analysed.
-   Limitations.

Pydantic can validate the response before the frontend renders it.

------------------------------------------------------------------------

## 10. Database design

Recommended core tables:

  Table            Purpose
  ---------------- -----------------------------------------------
  companies        Company metadata and aliases
  posts            Canonical social content
  post_analysis    Sentiment, topic, relevance and model outputs
  embeddings       Vector representations
  ingestion_runs   Collection history and errors

Possible future tables:

-   topic definitions.
-   aspect-level sentiment.
-   user queries.
-   AI responses.
-   model evaluation results.

### 10.1 Relationships

``` mermaid
erDiagram
    COMPANIES ||--o{ POSTS : tracks
    POSTS ||--o| POST_ANALYSIS : analysed_by
    POSTS ||--o{ EMBEDDINGS : represented_by
    INGESTION_RUNS ||--o{ POSTS : collects

    COMPANIES {
        int id PK
        string ticker
        string name
        string exchange
    }

    POSTS {
        int id PK
        int company_id FK
        string platform
        string external_id
        text content
        timestamp created_at
        json metadata
    }

    POST_ANALYSIS {
        int id PK
        int post_id FK
        string sentiment
        float confidence
        string topic
        float relevance_score
    }

    EMBEDDINGS {
        int id PK
        int post_id FK
        vector embedding
    }

    INGESTION_RUNS {
        int id PK
        string platform
        string status
        timestamp started_at
        timestamp completed_at
    }
```

### 10.2 Indexes

Consider:

-   Unique index on platform + external ID.
-   Index on company ID.
-   Index on publication timestamp.
-   Composite index on company ID + publication timestamp.
-   Index on platform and sentiment.
-   Vector index for similarity retrieval.

Avoid indiscriminate indexing: indexes increase storage and write
overhead.

------------------------------------------------------------------------

## 11. Frontend and user experience

### 11.1 Company overview

Provide:

-   Company selector.
-   Date range.
-   Total relevant posts.
-   Positive, neutral and negative distribution.
-   Sentiment trend.
-   Topic ranking.
-   Platform distribution.
-   Emerging concerns.
-   Data freshness.

### 11.2 Topic intelligence

Each topic should show:

-   Topic name.
-   Mention volume.
-   Sentiment distribution.
-   Change from the previous period.
-   Representative posts.
-   Source links.

### 11.3 Content explorer

Allow filtering by:

-   Company.
-   Platform.
-   Sentiment.
-   Topic.
-   Date.
-   Engagement.
-   Relevance.

Every post should link to its original source. Users must be able to
verify the model's interpretation.

### 11.4 AI assistant

Include:

-   Suggested questions.
-   Conversation history.
-   Evidence citations.
-   Source links.
-   Date and platform filters.
-   Clear indication of the dataset scope.

------------------------------------------------------------------------

## 12. Evaluation and measurement

The team has explicitly emphasised measurement. Do not merely claim that
the system works. Create small, specific tests and report actual
results.

### 12.1 Ingestion evaluation

  Metric                    Meaning
  ------------------------- ------------------------------------------
  Collection success rate   Successful API requests
  Ingestion completeness    Expected records successfully collected
  Duplicate rate            Duplicate records detected
  API failure rate          Failed requests
  Processing latency        Time from collection to availability
  Source coverage           Number of sources successfully collected

Test each adapter with a small bounded collection. For example, collect
up to 50 records from each platform where access permits and validate
the canonical schema.

### 12.2 Entity resolution

Create a manually labelled dataset containing relevant, irrelevant and
ambiguous posts.

Measure:

-   Precision.
-   Recall.
-   F1-score.

Precision reduces irrelevant content. Recall reduces missed investor
discussions. For an intelligence product, recall matters, but
low-confidence content should be clearly handled rather than silently
included.

### 12.3 Sentiment evaluation

Manually label approximately 100--200 posts if time permits.

Measure:

-   Accuracy.
-   Per-class precision and recall.
-   Macro F1.
-   Confusion matrix.

Macro F1 is important when classes are imbalanced. A model that predicts
the majority neutral class can have high accuracy while failing to
identify negative concerns.

### 12.4 RAG evaluation

Evaluate retrieval and generation separately.

**Retrieval:** - Precision@K. - Recall@K. - MRR. - NDCG@K where ranking
quality matters.

**Generation:** - Faithfulness to evidence. - Citation correctness. -
Answer relevance. - Completeness. - Correct abstention when evidence is
insufficient.

Prepare a small set of representative questions and manually identify
relevant evidence. A carefully evaluated set of 20 questions is more
valuable than an unsupported claim that the RAG system is accurate.

### 12.5 Performance and cost

Measure:

-   Ingestion runtime.
-   Classification latency.
-   Embedding throughput.
-   Retrieval latency.
-   LLM response latency.
-   Database query latency.
-   Cost per 1,000 processed posts.

Report only measurements obtained from your actual implementation.

------------------------------------------------------------------------

## 13. Production readiness and scaling

### 13.1 Growth scenario

If the system grows to 100,000 posts per day, evolve the MVP:

  MVP                     Possible future architecture
  ----------------------- ----------------------------------------
  APScheduler             Airflow or Prefect
  Direct API calls        Background ingestion workers
  In-process processing   Celery workers and message broker
  PostgreSQL              PostgreSQL plus object storage
  pgvector                Dedicated vector database if justified
  Streamlit               React or Next.js
  Modular monolith        Independently scalable services
  Local logs              Centralised observability

Do not add Kafka, Kubernetes or microservices merely to make the project
appear sophisticated.

### 13.2 Asynchronous processing

A future processing flow:

``` text
Ingestion API
     |
Message queue (Redis / RabbitMQ)
     |
+----+----------+-----------+
|               |           |
Worker 1      Worker 2    Worker 3
Sentiment     Topics      Embeddings
|               |           |
+---------------+-----------+
                |
        Database / vector index
```

This allows workers to scale independently and prevents long processing
jobs from blocking API requests.

### 13.3 Observability

Add:

-   Structured logs.
-   Error tracking.
-   API request metrics.
-   Source health checks.
-   Model latency tracking.
-   Token and cost tracking.
-   Failed-task retry monitoring.
-   Data freshness monitoring.

Data freshness is particularly important: the user should know when each
platform was last successfully collected.

### 13.4 Security and governance

Consider:

-   Secrets in environment variables or a secret manager.
-   No credentials committed to Git.
-   Input validation.
-   Parameterised SQL.
-   Rate limiting on chat endpoints.
-   Authentication and role-based access in a commercial deployment.
-   Tenant/company data isolation.
-   Safe handling of retrieved untrusted text.
-   Platform-specific retention and deletion policies.

------------------------------------------------------------------------

## 14. Important architectural trade-offs

  -----------------------------------------------------------------------
  Decision          MVP choice        Reason            When to
                                                        reconsider
  ----------------- ----------------- ----------------- -----------------
  Monolith vs       Modular monolith  Faster            Independent
  microservices                       development and   scaling or team
                                      simpler           boundaries
                                      operations        

  Batch vs          Scheduled batch   Sufficient for    Near-real-time
  streaming                           periodic          requirements
                                      intelligence      

  SQL vs NoSQL      PostgreSQL        Relational        Document
                                      analytics and     workloads become
                                      filtering         dominant

  pgvector vs       pgvector          One database and  Vector workload
  external vector                     simpler           justifies
  DB                                  deployment        separation

  Fine-tuning vs    Pretrained models No large labelled Measured
  prompting         and prompts       dataset yet       task-specific
                                                        performance gap

  Vector vs hybrid  Hybrid            Exact matching    Retrieval
  retrieval                           plus semantic     evaluation
                                      recall            indicates another
                                                        strategy

  One model vs      Task-specific     Cost and          Operational
  multiple models   models            performance       complexity
                                      differ by task    outweighs
                                                        benefits

  Agentic vs        Deterministic     Predictability    Complex tool
  deterministic     query planner     and auditability  selection is
  workflow                                              demonstrably
                                                        useful

  LLM vs SQL for    SQL               Deterministic     Do not replace
  metrics                             numerical answers SQL with an LLM
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 15. Product thinking: turn sentiment into investor intelligence

A simple sentiment dashboard is insufficient.

Consider a hypothetical company discussion:

-   Revenue growth: positive.
-   Debt: negative.
-   Network expansion: positive.
-   Regulatory concerns: mixed.

Overall sentiment may be positive, while debt-related discussion is
increasing. The IR team needs to investigate that specific concern.

### 15.1 Separate opinion from fact

If a user posts "the company is going bankrupt", the application should
report that a user expressed concern about financial stability. It must
not report bankruptcy as an established fact.

The product monitors public conversation; it does not independently
verify every claim.

### 15.2 Show coverage and confidence

Always show:

-   Number of posts analysed.
-   Date range.
-   Platforms included.
-   Data freshness.
-   Model confidence or evaluation caveats.
-   Sample-size warnings.

Public social content is not a statistically representative sample of
all shareholders. Do not imply that social sentiment represents the
entire investor population.

### 15.3 Emerging concern prioritisation

A future Investor Concern Score could combine:

\[ ICS = `\alpha `{=tex}V + `\beta `{=tex}G + `\gamma `{=tex}N +
`\delta `{=tex}P \]

Where:

-   (V): Normalised discussion volume.
-   (G): Discussion growth.
-   (N): Negative sentiment proportion.
-   (P): Platform breadth.

Weights must be validated. This is a prioritisation aid for IR teams,
not a financial risk score or stock-price prediction.

For the MVP, show the component measures separately rather than
presenting an unvalidated composite score.

------------------------------------------------------------------------

## 16. Three-hour implementation plan

The architecture above is the target design. The actual task should be a
tightly scoped vertical slice.

  -----------------------------------------------------------------------
  Time                    Work                    Deliverable
  ----------------------- ----------------------- -----------------------
  0--15 min               Requirements, schemas   Defined scope
                          and architecture        

  15--50 min              Three ingestion         Normalised source
                          adapters                records

  50--75 min              Database and processing Stored, deduplicated,
                                                  enriched posts

  75--100 min             Analytics               Sentiment and topic
                                                  aggregates

  100--135 min            RAG                     Filtered retrieval and
                                                  cited answers

  135--160 min            Frontend                Overview, explorer and
                                                  assistant

  160--175 min            Testing                 Ingestion,
                                                  classification and
                                                  retrieval checks

  175--180 min            Documentation           Run instructions and
                                                  limitations
  -----------------------------------------------------------------------

### Phase 1: Ingestion

1.  Build one company first.
2.  Implement Bluesky, YouTube and Reddit adapters.
3.  Normalise into one schema.
4.  Persist records.
5.  Handle pagination, retries and deduplication.

If an API credential or permission is unavailable, implement the adapter
boundary and document the limitation. Use clearly labelled sample
fixtures only where necessary.

### Phase 2: Processing and analytics

Implement:

-   Normalisation.
-   Deduplication.
-   Entity matching.
-   Sentiment classification.
-   Topic classification.
-   Engagement metrics.

Avoid training a custom model during the short build.

### Phase 3: Dashboard

Build:

-   Company selector.
-   Date filter.
-   Sentiment chart.
-   Topic breakdown.
-   Platform comparison.
-   Post explorer.

Use real database aggregations.

### Phase 4: RAG

Implement:

1.  Embedding generation.
2.  Vector storage.
3.  Metadata filtering.
4.  Similarity and keyword retrieval.
5.  Context construction.
6.  LLM generation.
7.  Citation validation.

Avoid multi-agent orchestration.

### Phase 5: Validation and documentation

Verify the ingestion paths, company filters, sentiment outputs,
citations, empty-result behaviour and Docker Compose setup. Document
what is real, synthetic, incomplete or limited.

------------------------------------------------------------------------

## 17. Five-minute demo plan

1.  **Problem:** explain that IR teams currently monitor multiple
    platforms manually.
2.  **Ingestion:** show the three adapters, collection status and
    canonical schema.
3.  **Company intelligence:** select VEON and show discussion volume,
    sentiment, topics and platform comparison.
4.  **Evidence:** click a negative topic and inspect original posts.
5.  **AI assistant:** ask "What are the three biggest concerns investors
    have about VEON, and what evidence supports them?"
6.  **Evaluation:** show the sentiment test set, retrieval checks and
    actual measured results.

Suggested questions:

-   What are the most discussed topics about VEON?
-   What are investors most concerned about?
-   Compare negative sentiment across all three platforms.
-   What changed over the last seven days?
-   Show evidence for the biggest concern.
-   What are investors saying about TTWO's GTA VI release?

For the last question, distinguish gaming enthusiasm from
investment-related discussion.

------------------------------------------------------------------------

## 18. Likely interview questions

### Architecture

**Why these three platforms?**\
Discuss relevance, content diversity, API access, cost, terms and
implementation time.

**What happens if one platform API goes down?**\
Independent adapters, retries, partial success, source health tracking
and data freshness indicators.

**How do you prevent duplicates?**\
Platform ID uniqueness, idempotent ingestion, unique constraints and
normalised text hashes.

**How would you scale to millions of posts?**\
Queues, asynchronous workers, database indexing, batch embeddings,
observability and independent scaling.

**Why PostgreSQL instead of a vector database?**\
The MVP needs structured analytics and semantic search; pgvector avoids
premature infrastructure complexity.

### Machine learning

**How do you handle sarcasm?**\
Acknowledge model limitations, label sarcasm in an evaluation set and
improve with domain-specific training data.

**How do you handle class imbalance?**\
Macro F1, per-class precision and recall, stratified evaluation and
class-weighted training if fine-tuning is justified.

**What if sentiment is wrong?**\
Store confidence and model version, evaluate against human labels,
preserve source text and support reprocessing.

**How do you detect emerging topics?**\
Volume baselines, rolling windows, clustering and change detection.

**How do you prevent engagement bias?**\
Show post-weighted and engagement-weighted sentiment separately; use
post-weighted sentiment as the default.

### RAG

**Why RAG instead of fine-tuning?**\
RAG provides access to newly collected data without retraining and
supports evidence citations.

**Why hybrid retrieval?**\
Keyword search captures exact terms; vector search captures semantic
similarity; metadata filters enforce scope.

**How do you prevent hallucinations?**\
Filtered retrieval, evidence-grounded prompts, structured output,
citation validation and abstention.

**How do you answer numerical questions?**\
Use SQL aggregation and let the LLM explain the computed results.

**How do you evaluate RAG?**\
Evaluate retrieval separately from generation using Precision@K,
Recall@K, faithfulness and citation correctness.

### Product

**What is the most valuable feature?**\
Evidence-backed investor concern detection, because it helps IR teams
identify and investigate important themes.

**What would you build next?**\
Emerging-concern alerts, official-disclosure comparison and better
historical trend analysis.

**What are the biggest risks?**\
Platform access and licensing, noisy data, entity misclassification,
sentiment errors, unrepresentative discussions, hallucinations and stale
data.

------------------------------------------------------------------------

## 19. What to leave out of the MVP

  -----------------------------------------------------------------------
  Feature                             Reason to defer
  ----------------------------------- -----------------------------------
  Real-time streaming                 Not needed to validate the core
                                      product

  Custom fine-tuned models            No sufficiently large labelled
                                      dataset

  Complex multi-agent system          More complexity than value

  Kubernetes                          Unnecessary infrastructure at this
                                      scale

  Kafka                               Batch ingestion is sufficient

  Multiple vector databases           pgvector is sufficient initially

  Stock-price prediction              Outside the core product objective

  Financial recommendation engine     Not required

  Definitive bot identification       Requires stronger evidence

  Automated IR responses              Requires human approval and
                                      governance

  Full multilingual support           Start with a defined supported
                                      language set
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 20. README contents

The submitted README should contain:

1.  Project overview.
2.  Business problem.
3.  Architecture diagram.
4.  Technology choices and justification.
5.  Supported companies and platform integrations.
6.  Data model.
7.  Ingestion methodology.
8.  ML and analytics methodology.
9.  RAG architecture.
10. Evaluation results.
11. Installation instructions.
12. Environment variables.
13. Docker execution instructions.
14. Known limitations.
15. Future improvements.

Be explicit about:

-   Which integrations use real APIs.
-   Which records are synthetic.
-   How much data was collected.
-   Supported languages.
-   Actual sentiment evaluation results.
-   Historical coverage.
-   Required API permissions.
-   Data retention limitations.

Never present synthetic demonstration data as real public social
content.

------------------------------------------------------------------------

## 21. Final implementation priorities

  Priority   Deliverable                                  Importance
  ---------- -------------------------------------------- ------------
  P0         Three ingestion adapters and common schema   Critical
  P0         Company-specific data separation             Critical
  P0         Sentiment and topic analytics                Critical
  P0         Working RAG with citations                   Critical
  P0         Usable dashboard                             Critical
  P1         Evaluation and testing                       Very high
  P1         Docker and README                            Very high
  P1         Data freshness and ingestion monitoring      High
  P2         Aspect-based sentiment                       Valuable
  P2         Emerging topic detection                     Valuable
  P2         Hybrid retrieval and reranking               Valuable
  P3         Kafka, Kubernetes and microservices          Future
  P3         Fine-tuned models and agentic workflows      Future

Build one complete vertical slice:

**One company → three platforms → one intelligence pipeline → one
dashboard → one grounded AI answer.**

Then extend the same implementation to STB and TTWO.

------------------------------------------------------------------------

## 22. Suggested interview positioning

> I approached this as an end-to-end investor intelligence product
> rather than a conventional sentiment analysis application.
>
> The system collects public discussions from three social platforms
> through independent ingestion adapters. I normalise the different
> source formats into a common data model and enrich the content with
> company relevance, sentiment and topic information.
>
> I use PostgreSQL for structured analytics and pgvector for semantic
> retrieval. The AI assistant uses a hybrid RAG architecture, combining
> structured database queries with semantic retrieval so numerical
> questions are answered deterministically and qualitative questions are
> grounded in actual social content.
>
> The frontend is designed around the IR team's workflow: understanding
> the overall conversation, identifying emerging concerns, exploring
> supporting posts and asking questions through an evidence-backed AI
> assistant.
>
> I have deliberately kept the MVP architecture simple, with clear
> module boundaries, while identifying how ingestion, processing and
> retrieval could scale independently in a production environment.

## 23. Five flows to explain without looking at the code

1.  **Data flow:** how a Reddit post becomes a database record.
2.  **ML flow:** how the record gets relevance, sentiment and topic
    labels.
3.  **Retrieval flow:** how a user question becomes filtered keyword and
    vector search.
4.  **Generation flow:** how retrieved posts become a cited, grounded
    answer.
5.  **Scaling flow:** how the application could process 100,000 posts
    per day without blocking the user-facing application.

The strongest interview performance will come from explaining these
flows, the trade-offs, the failure modes and the measurements---not from
adding the largest number of technologies.



Built complete and after building it teach me