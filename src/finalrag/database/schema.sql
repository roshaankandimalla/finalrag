CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    document_id UUID PRIMARY KEY,
    domain TEXT NOT NULL,
    file_name TEXT NOT NULL,
    source_type TEXT NOT NULL,
    file_path TEXT NOT NULL,
    parser_used TEXT,
    parser_version TEXT,
    status TEXT NOT NULL DEFAULT 'discovered',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS parent_sections (
    parent_id UUID PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES documents(document_id)
        ON DELETE CASCADE,
    domain TEXT NOT NULL,
    source_type TEXT NOT NULL,
    section_title TEXT,
    section_path TEXT[],
    page_numbers INTEGER[],
    row_ranges JSONB,
    parent_text TEXT,
    token_count INTEGER,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS child_chunks (
    chunk_id UUID PRIMARY KEY,
    parent_id UUID REFERENCES parent_sections(parent_id)
        ON DELETE CASCADE,
    document_id UUID NOT NULL REFERENCES documents(document_id)
        ON DELETE CASCADE,
    domain TEXT NOT NULL,
    source_type TEXT NOT NULL,
    file_name TEXT NOT NULL,
    retrieval_type TEXT NOT NULL DEFAULT 'child_chunk',
    retrieval_text TEXT,
    page_numbers INTEGER[],
    source_urls TEXT[],
    source_row_numbers BIGINT[],
    row_range JSONB,
    section_title TEXT,
    section_path TEXT[],
    modalities TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    text_content TEXT,
    table_markdown TEXT,
    table_html TEXT,
    image_paths TEXT[],
    token_count INTEGER,
    embedding_input_hash TEXT,
    dense_embedding VECTOR(2048),
    dense_model TEXT,
    dense_dimension INTEGER,
    dense_updated_at TIMESTAMPTZ,
    sparse_embedding SPARSEVEC(30522),
    sparse_model TEXT,
    sparse_dimension INTEGER,
    sparse_updated_at TIMESTAMPTZ,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS domain_centroids (
    centroid_id TEXT PRIMARY KEY,
    domain TEXT NOT NULL,
    retrieval_type TEXT NOT NULL,
    chunk_count BIGINT NOT NULL,
    dense_model TEXT NOT NULL,
    dense_dimension INTEGER NOT NULL,
    centroid_embedding VECTOR(2048) NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (domain, retrieval_type, dense_model, dense_dimension)
);

-- Keep schema.sql safe to rerun against databases created by earlier versions.
ALTER TABLE parent_sections
    ADD COLUMN IF NOT EXISTS token_count INTEGER,
    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();

ALTER TABLE child_chunks
    ADD COLUMN IF NOT EXISTS retrieval_type TEXT NOT NULL DEFAULT 'child_chunk',
    ADD COLUMN IF NOT EXISTS retrieval_text TEXT,
    ADD COLUMN IF NOT EXISTS source_urls TEXT[],
    ADD COLUMN IF NOT EXISTS source_row_numbers BIGINT[],
    ADD COLUMN IF NOT EXISTS section_path TEXT[],
    ADD COLUMN IF NOT EXISTS token_count INTEGER,
    ADD COLUMN IF NOT EXISTS embedding_input_hash TEXT,
    ADD COLUMN IF NOT EXISTS dense_embedding VECTOR(2048),
    ADD COLUMN IF NOT EXISTS dense_model TEXT,
    ADD COLUMN IF NOT EXISTS dense_dimension INTEGER,
    ADD COLUMN IF NOT EXISTS dense_updated_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS sparse_embedding SPARSEVEC(30522),
    ADD COLUMN IF NOT EXISTS sparse_model TEXT,
    ADD COLUMN IF NOT EXISTS sparse_dimension INTEGER,
    ADD COLUMN IF NOT EXISTS sparse_updated_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();

DO $$
DECLARE
    dense_column_type TEXT;
BEGIN
    SELECT format_type(attribute.atttypid, attribute.atttypmod)
    INTO dense_column_type
    FROM pg_attribute AS attribute
    WHERE attribute.attrelid = 'child_chunks'::regclass
      AND attribute.attname = 'dense_embedding'
      AND NOT attribute.attisdropped;

    IF dense_column_type IS DISTINCT FROM 'vector(2048)' THEN
        IF EXISTS (
            SELECT 1
            FROM child_chunks
            WHERE dense_embedding IS NOT NULL
        ) THEN
            RAISE EXCEPTION
                'Cannot migrate dense_embedding from % to vector(2048): embeddings already exist',
                dense_column_type;
        END IF;

        DROP INDEX IF EXISTS idx_child_chunks_dense_embedding;
        ALTER TABLE child_chunks
            ALTER COLUMN dense_embedding TYPE VECTOR(2048)
            USING NULL::VECTOR(2048);
    END IF;
END
$$;

CREATE TABLE IF NOT EXISTS hcahps_records (
    record_id BIGSERIAL PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES documents(document_id)
        ON DELETE CASCADE,
    source_row_number BIGINT NOT NULL,
    facility_id TEXT NOT NULL,
    facility_name TEXT NOT NULL,
    address TEXT,
    city TEXT,
    state TEXT,
    zip_code TEXT,
    county TEXT,
    telephone TEXT,
    measure_id TEXT NOT NULL,
    question TEXT,
    answer_description TEXT,
    star_rating TEXT,
    star_rating_footnote TEXT,
    answer_percent TEXT,
    answer_percent_footnote TEXT,
    linear_mean_value TEXT,
    completed_surveys TEXT,
    completed_surveys_footnote TEXT,
    response_rate_percent TEXT,
    response_rate_footnote TEXT,
    survey_start_date DATE,
    survey_end_date DATE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (document_id, source_row_number)
);

CREATE TABLE IF NOT EXISTS hospital_profiles (
    profile_id UUID PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES documents(document_id)
        ON DELETE CASCADE,
    facility_id TEXT NOT NULL,
    hospital_name TEXT NOT NULL,
    address TEXT,
    city TEXT,
    state TEXT,
    zip_code TEXT,
    county TEXT,
    telephone TEXT,
    survey_start_date DATE,
    survey_end_date DATE,
    completed_surveys TEXT,
    response_rate_percent TEXT,
    category_summaries JSONB NOT NULL DEFAULT '{}'::jsonb,
    retrieval_text TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (document_id, facility_id)
);

CREATE TABLE IF NOT EXISTS hospital_category_docs (
    category_doc_id UUID PRIMARY KEY,
    profile_id UUID NOT NULL REFERENCES hospital_profiles(profile_id)
        ON DELETE CASCADE,
    document_id UUID NOT NULL REFERENCES documents(document_id)
        ON DELETE CASCADE,
    facility_id TEXT NOT NULL,
    category TEXT NOT NULL,
    retrieval_text TEXT NOT NULL,
    table_markdown TEXT NOT NULL,
    measure_ids TEXT[] NOT NULL,
    source_row_numbers BIGINT[] NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (document_id, facility_id, category)
);

CREATE INDEX IF NOT EXISTS idx_documents_domain
    ON documents(domain);

CREATE INDEX IF NOT EXISTS idx_documents_status
    ON documents(status);

CREATE INDEX IF NOT EXISTS idx_parent_sections_document
    ON parent_sections(document_id);

CREATE INDEX IF NOT EXISTS idx_child_chunks_document
    ON child_chunks(document_id);

CREATE INDEX IF NOT EXISTS idx_child_chunks_parent
    ON child_chunks(parent_id);

CREATE INDEX IF NOT EXISTS idx_child_chunks_domain
    ON child_chunks(domain);

CREATE INDEX IF NOT EXISTS idx_child_chunks_dense_pending
    ON child_chunks(dense_model, dense_dimension)
    WHERE dense_embedding IS NULL;

CREATE INDEX IF NOT EXISTS idx_child_chunks_sparse_pending
    ON child_chunks(sparse_model, sparse_dimension)
    WHERE sparse_embedding IS NULL;

CREATE INDEX IF NOT EXISTS idx_domain_centroids_dense_embedding
    ON domain_centroids USING hnsw (
        (centroid_embedding::halfvec(2048)) halfvec_cosine_ops
    );

CREATE INDEX IF NOT EXISTS idx_hcahps_records_document
    ON hcahps_records(document_id);

CREATE INDEX IF NOT EXISTS idx_hcahps_records_facility
    ON hcahps_records(facility_id);

CREATE INDEX IF NOT EXISTS idx_hcahps_records_state
    ON hcahps_records(state);

CREATE INDEX IF NOT EXISTS idx_hcahps_records_measure
    ON hcahps_records(measure_id);

CREATE INDEX IF NOT EXISTS idx_hospital_profiles_document
    ON hospital_profiles(document_id);

CREATE INDEX IF NOT EXISTS idx_hospital_profiles_state
    ON hospital_profiles(state);

CREATE INDEX IF NOT EXISTS idx_hospital_category_docs_document
    ON hospital_category_docs(document_id);

CREATE INDEX IF NOT EXISTS idx_hospital_category_docs_facility
    ON hospital_category_docs(facility_id);

CREATE INDEX IF NOT EXISTS idx_hospital_category_docs_category
    ON hospital_category_docs(category);


    
