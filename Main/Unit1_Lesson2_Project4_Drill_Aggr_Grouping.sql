-- Drill: Aggregating and grouping

-- What was the hottest day in our data set? Where was that?
SELECT
	--maxtemperaturef,
	MAX(maxtemperaturef) as maxtemp,
	zip
FROM
	weather
GROUP BY
	zip
ORDER BY maxtemp DESC
LIMIT 1

-- How many trips started at each station?
SELECT
	count(trip_id) as amount_of_trips,
	start_station
FROM
	trips
GROUP BY
	start_station
	
-- What's the shortest trip that happened?
SELECT
	trip_id,
	MIN(duration) as min_duration
FROM
	trips
GROUP BY
	trip_id
ORDER BY min_duration ASC

-- What is the average trip duration, by end station?
SELECT
	end_station,
	AVG(duration) as avg_duration
FROM
	trips
GROUP BY
	end_station