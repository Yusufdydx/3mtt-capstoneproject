# SME Demand Forecasting for Mr Joseph's FMCG Distribution Business

## The Problem

Mr Joseph is a Nigerian entrepreneur who distributes FMCG products, milk, yogurt, ready meals, juice, and snack bars, across Nigeria. He supplies through three channels, his own Retail counters and partner shops, Discount resellers who buy in bulk, and E-commerce orders placed online or through WhatsApp. He operates across three regions of the country.

Like most distributors, Mr Joseph faces a constant balancing act. Order too little stock and he risks running out and losing sales. Order too much and his capital sits tied up on a shelf instead of working for his business. Right now that balancing act runs on instinct and experience. This project builds a data driven tool to help make it more precise.

## What This Project Does

From the sample datasets from Kaggle via (https://www.kaggle.com/datasets/beatafaron/fmcg-daily-sales-data-to-2022-2024?hl=en-US), we start by using close to three years of daily sales history, January 2022 through December 2024, across 30 products, this project builds a system that:

1. **Forecasts how much of each product will sell**, for each product, each sales channel, and each region, on any given day.
2. **Flags which specific products are at risk of running out**, or are sitting with more stock than they actually need, right now.
3. **Points to exactly where Mr Joseph should focus his attention first**, backed by evidence, not guesswork.

## A Quick Note on the Data

The dataset behind this project is synthetic, meaning it was generated to behave like real FMCG sales data rather than pulled directly from an actual company's records. It is not from a real named business. Before being used for this project, it was checked carefully and confirmed to show the kind of patterns real sales data actually has, a genuine sales trend, a real effect from promotions, and real, observable stockouts, rather than just random noise. Wherever the data has a real limitation, this project reports it honestly rather than hiding it, in the same spirit as everything else in this README.

## What the Forecast Actually Achieves

The forecasting model was tested against the simplest possible alternative, just guessing that tomorrow's sales will match yesterday's. Compared to that simple guess, the model cuts the average forecasting error by roughly **60 percent**. In plain terms, on a typical day the model's forecast is off by about 3 units per product, channel, and region combination, compared to being off by about 9 units if Mr Joseph were just going on yesterday's number alone.

The two things that matter most to the forecast, by far, are **how much stock is currently available** and **whether a promotion is running**. That matches common sense, you cannot sell what you do not have on the shelf, and promotions genuinely move product, roughly doubling average daily sales when one is active.

## The Inventory Risk Tool

For every product, channel, and region combination, the system compares current stock against forecasted demand to calculate a **days of cover** number, essentially, how many more days would this stock last at the expected rate of sale. That number sorts every combination into one of four categories:

- **Stockout** — already out of stock, right now.
- **At Risk** — stock will likely run out before a new delivery could realistically arrive, based on Mr Joseph's own historical delivery times.
- **Healthy** — stock levels are appropriately matched to expected demand.
- **Overstocked** — stock is sitting well beyond what is actually needed, tying up capital that could be used elsewhere.

Across a recent two month period used to test this system, the picture looks like this:

| Status | Share of Inventory |
|---|---|
| Healthy | 81.8% |
| Overstocked | 12.7% |
| At Risk | 3.5% |
| Stockout | 2.0% |

## Where to Focus First

The single most useful, specific finding from this whole project is at the product category level, not the channel or region level. Channel (Retail, Discount, E-commerce) and region (North, Central, South) turned out to carry very similar risk levels across the board, no one channel or region stands out as a clear problem area.

Category is a different story:

- **Ready Meals carry by far the most stockout risk**, nearly 12 percent of the time this category is flagged as at risk of running out, more than three times any other category. This appears to be because Ready Meal demand is genuinely less predictable day to day than other products, making it harder to keep reliably stocked.
- **Juice is significantly overstocked**, 40 percent of the time it is flagged as carrying more stock than recent demand justifies. Juice is Mr Joseph's smallest line, with only one product, so there is no group of similar products to smooth out its demand pattern, and stock levels appear to have been set higher than current sales actually need.

**Concrete recommendation:** Mr Joseph should prioritize tightening reorder timing on his Ready Meal line first, since that is where real stockout risk is concentrated, and can likely reduce how much Juice stock he keeps on hand without meaningfully increasing his risk of running out.

## Honest Limitations

This project reports its limitations plainly rather than glossing over them:

- The inventory risk flag is a reliable snapshot of stock health **right now**, but testing showed it is **not yet a strong early warning signal** for stockouts a few days in advance, in this dataset deliveries do not appear to arrive in direct response to how low stock currently is, so today's stock level does not reliably predict whether a delivery will arrive in time.
- The forecast tends to run about 4.6 percent higher than actual demand in aggregate, a small but real bias, in the safer direction for planning purposes since underestimating demand is the more costly mistake.
- The model handles average promotion effects well but can underestimate unusually large promotional spikes, since it only knows whether a promotion is happening, not how big it is.

## What's Included

- **Part 1, Data Understanding** — loading and verifying the dataset, explaining every column in Mr Joseph's business context, and relabeling the dataset's original regional codes to fit his Nigerian operations.
- **Part 2, Exploratory Data Analysis** — testing every assumption about the data honestly against real evidence.
- **Part 3, Model Development** — building and tuning the forecasting model, including catching and properly fixing a real data quality issue found partway through.
- **Part 4, Inventory Risk Classification** — building and testing the days of cover system.
- **Part 5, Results and Communication** — final charts, model evaluation, and this business narrative.

## Tools Used

Python, pandas, and scikit learn, run in Google Colab. No deep learning or heavy frameworks were used, in line with this being a minimum viable product rather than a research exercise.

## Dataset Source:
Kaggle: https://www.kaggle.com/datasets/beatafaron/fmcg-daily-sales-data-to-2022-2024?hl=en-US

