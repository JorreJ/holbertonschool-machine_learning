-- Display the maximum temperature recorded for each state, ordered alphabetically by state
SELECT state, MAX(value) AS max_temp
FROM temperatures
GROUP BY state
ORDER BY state;
