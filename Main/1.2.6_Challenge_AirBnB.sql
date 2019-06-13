-- Challenge: AirBnB in SanFran

-- What's the most expensive listing? What else can you tell me about the listing?
SELECT 
	*
FROM public.sfo_listings
ORDER BY price DESC
LIMIT 1;
-- Result: "ID: 24650875 - Full House Victorian: 7500 SqFt, 4 Floors, Hot Tub - Entire home/apt"

-- What neighborhoods seem to be the most popular?
SELECT
	neighbourhood,
	AVG(reviews_per_month) AS avg_review_score
FROM
	sfo_listings
GROUP BY neighbourhood
ORDER BY avg_review_score DESC
LIMIT 1;
-- Result: "Presidio"

-- What time of year is the cheapest time to go to San Francisco? 
SELECT
	EXTRACT(MONTH FROM calender_date),
	AVG(price::money::numeric) AS avg_price
FROM
	sfo_calendar
GROUP BY 1
ORDER BY avg_price ASC
-- Result: 1st Jan, 2nd Feb, 3rd Dec
-- What about the busiest?
-- Result: September