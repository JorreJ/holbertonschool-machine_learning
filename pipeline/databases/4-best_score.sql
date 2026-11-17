-- Retrieve the name and score of records with a score of 10 or higher,
-- ordering the results by score in descending order (highest to lowest).
SELECT score, name
FROM second_table
WHERE score >= 10
ORDER BY score DESC;
