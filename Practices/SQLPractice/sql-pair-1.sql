select * from names limit 10;
-- What does each row of data represent in this table?
-- represent one person with his name, gender, state and frequency.

-- Show only the name and frequency fields for the first 5 records.
select name , frequency from names limit 5 ;


-- Find records for John's born in Washington state after 2010.
select * from names where name = 'John' and state = 'WA' and year > 2010

-- How many John's are there in the dataset?
select sum(frequency)  from names where name = 'John' ;

-- Show situations where girls were named John. In which states and years did this happen the most?
select * from names where gender = 'F' and name = 'John'  order by frequency desc limit 1;


-- What were the top 3 most common female names in New York in the year 2000?
select name ,frequency
from names
where
      gender = 'F'
  and state = 'NY'
  and year = 2000
order by frequency DESC
limit 3;


-- How many babies are born each year?
select year ,  sum(frequency)
from names
group by year ;


-- How many John's were there per state, per year?
select state , year, sum(frequency)
from names
where name = 'John'
group by state , year;


-- Write a query that tells you how many different female names there were per state, per year.
select state , year, count(name)
from names
where gender = 'F'
group by state , year;


-- How many records were there in the years 2000, 2001 and 2002?
select count(*)
from names
where year between 2000 and 2002;


-- How many records were there in the years 2000, 2001 and 2002?
select count(*)
from names
where year in (2000 , 2001, 2002) ;

-- How many names end with the letter ‘a’ in the table? Answer: 7,608
select  count(distinct (name))
from names
where name like '%a' ;


-- What are the columns on the region table?
PRAGMA table_info(regions) ;

-- How many different regions are there in the region table? (Hint: DISTINCT) Can you find the one that looks like a typo?
select distinct (region)
from regions;


-- to show all tables in DataBase
select name
from sqlite_master
where type = 'table';

