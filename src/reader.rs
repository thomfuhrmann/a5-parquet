use std::error::Error;

use datafusion::{arrow::util::pretty::pretty_format_batches, prelude::*};

pub async fn read_parquet() -> Result<(), Box<dyn Error>> {
    let ctx = SessionContext::new();
    ctx.register_parquet(
        "countries",
        "tests/output/countries.parquet",
        ParquetReadOptions::new(),
    )
    .await?;

    // create a plan
    let df = ctx.sql("SELECT id, bbox FROM countries").await?;
    // let df = ctx
    //     .sql("SELECT id, bbox FROM countries WHERE bbox.xmin < 130 AND bbox.xmax > 145 AND bbox.ymin > 30 AND bbox.ymax > 45")
    //     .await?;

    // execute the plan
    let results = df.collect().await?;

    // format the results
    let pretty_results = pretty_format_batches(&results)?.to_string();
    println!("{pretty_results}");

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::reader::read_parquet;

    #[tokio::test]
    async fn test_read_parquet() {
        let _ = read_parquet().await;
    }
}
