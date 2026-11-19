-- Display the average temperature for each city, ordered from highest to lowest
SELECT City, AVG(value) AS avg_temp FROM temperatures
GROUP BY City
ORDER BY avg_temp DESC;
