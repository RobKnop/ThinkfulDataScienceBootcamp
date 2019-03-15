-- Drill: Joins and CTEs

-- What are the three longest trips on rainy days?
WITH
    rainy_days
AS (
    SELECT
        date
    FROM
        weather
    WHERE events LIKE 'Rain%'
)
SELECT
	DISTINCT(trps.trip_id),
	trps.duration,
	rd.date
FROM
	rainy_days as rd
INNER JOIN
	trips AS trps
ON
	DATE(trps.start_date) = DATE(rd.date) AND DATE(trps.end_date) = DATE(rd.date)
ORDER BY trps.duration DESC
LIMIT 3;

-- Which station is full most often?
WITH
	count_start_terminal
AS (
SELECT
	Count(trips.trip_id),
	start_terminal
FROM
	trips
GROUP BY start_terminal
),
	count_end_terminal
AS (
SELECT
	Count(trips.trip_id),
	end_terminal
FROM
	trips
GROUP BY end_terminal
)
SELECT
	cst.count + cet.count AS overall_count,
	cst.start_terminal,
	cet.end_terminal,
	stations.name
FROM
	count_start_terminal AS cst
JOIN
	count_end_terminal AS cet
ON
	cst.start_terminal = cet.end_Terminal
LEFT JOIN
	stations
ON
	cst.start_terminal = stations.station_id	
ORDER BY overall_count DESC;

-- Return a list of stations with a count of number of trips starting at that station but ordered by dock count.
WITH
	count_start_terminal
AS (
SELECT
	Count(trips.trip_id),
	start_terminal
FROM
	trips
GROUP BY start_terminal
)
SELECT
	cst.count AS count_of_starts,
	stations.name,
	stations.dockcount as dc
FROM
	count_start_terminal AS cst
RIGHT JOIN
	stations
ON
	cst.start_terminal = stations.station_id	
ORDER BY dc DESC;

-- (Challenge) What's the length of the longest trip for each day it rains anywhere?
WITH
    rainy_days
AS (
    SELECT
        date
    FROM
        weather
    WHERE events LIKE 'Rain%'
)
SELECT
	DISTINCT(trps.trip_id),
	trps.duration,
	rd.date
FROM
	rainy_days as rd
INNER JOIN
	trips AS trps
ON
	DATE(trps.start_date) = DATE(rd.date) OR DATE(trps.end_date) = DATE(rd.date)
ORDER BY trps.duration DESC
LIMIT 3;