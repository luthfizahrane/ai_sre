-- Run read-only BEFORE making any changes to an existing Supabase project.
-- Compare returned columns with migrations/001_canonical_schema.sql.
SELECT table_name, column_name, data_type, is_nullable
FROM information_schema.columns
WHERE table_schema = 'public'
  AND table_name IN ('document_chunks', 'nodes_kg', 'edges_kg')
ORDER BY table_name, ordinal_position;

-- After verifying that edges_kg has the canonical column names, optionally run:
-- SELECT "fromId", "toId", relation, count(*) AS duplicates
-- FROM public.edges_kg
-- GROUP BY "fromId", "toId", relation
-- HAVING count(*) > 1;
