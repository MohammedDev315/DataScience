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


create table RESTAURANTS
(
	Name varchar,
	Cuisine varchar,
	Date_Founded Date
);



insert into
    RESTAURANTS(Name , Cuisine , Date_Founded )
    values ( 'Chipotle' , 'Mexican' ,  7-13-1993 ) ;




-- Which state(s) does not have a region associated with it? Hint: remember DISTINCT Answer: MI
--
select distinct(n.state) , r.state
from names n
    left join  regions r
        on n.state = r.state
where r.state is null;


-- What is the most popular boy's name in the South in 2000? Hint: remember GROUP BY Answer: Jacob
--

select name , sum(frequency) as fre_total
from names n
    join regions r
        on n.state = r.state
where r.region = 'South' and n.year = 2000 and n.gender='M'
group by n.name
order by  fre_total desc ;






-- What is the third most popular girl’s name in the south in the year 2000? Answer: Madison
--
select name , sum(frequency) as fre_total
from names n
    join regions r
        on n.state = r.state
where r.region = 'South' and n.year = 2000 and n.gender='F'
group by n.name
order by  fre_total desc limit 3 ;

-- Which state has the largest number of unique names in the year 2000? Answer: CA
--
select   count(distinct(name)) , state
from names
where year = 2000
group by state
order by 1 desc ;


-- Which region has the largest number of unique names in the year 2000? Answer: Pacific
--

select  count(distinct(n.name)) , n.state , r.region
from names n  join regions r
    on n.state = r.state
where n.year = 2000
group by r.region
order by 1 desc ;


-- Write a query that shows the number of babies born in each region. Exclude the blank region.
select   r.region , n.year , sum(n.frequency)
from names n  left join regions r
    on n.state = r.state
where r.region is not null
group by r.region ;


-- select name
--     CASE name = 'ADD'
--     THEN 'GOOD'
--     END
--     AS New_colum
-- from names ;

--
-- select name
--     CASE
--         when name = '' or name = ''
--         then 'Wlecome'
--         else 'NEe'
--     end
--     as colum_name
-- from names ;

