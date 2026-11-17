-- Create the first_table table if it does not already exist.
-- The table contains an integer ID and a name (up to 256 characters).
CREATE TABLE IF NOT EXISTS first_table (
    id INT,
    name VARCHAR(256)
);
