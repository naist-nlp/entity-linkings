import argparse
import gzip
import io
import re

import duckdb
import pandas as pd
import requests


def process_and_insert(con: duckdb.DuckDBPyConnection, url: str, table_name: str, pattern: re.Pattern) -> None:
    total_records = 0
    batch_size = 100000
    batch = []

    print(f"\nStreaming from URL: {url}")
    with (
        requests.get(url, stream=True, timeout=(10, 300)) as response,
        gzip.GzipFile(fileobj=response.raw) as gz,
        io.TextIOWrapper(gz, encoding='utf-8', errors='ignore') as f
    ):
        response.raise_for_status()
        for line in f:
            if line.startswith("INSERT INTO"):
                matches = pattern.findall(line)
                for m in matches:
                    processed_row = (int(m[0]), m[1].replace("\\'", "'"))
                    batch.append(processed_row)

                if len(batch) >= batch_size:
                    df_batch = pd.DataFrame(batch, columns=['page_id', 'title'])
                    con.execute(f"INSERT INTO {table_name} SELECT * FROM df_batch")
                    total_records += len(batch)
                    batch = []
                    print(f"Now inserting to {table_name}... Current total records: {total_records}")

    if batch:
        df_batch = pd.DataFrame(batch, columns=['page_id', 'title'])
        con.execute(f"INSERT INTO {table_name} SELECT * FROM df_batch")
        total_records += len(batch)
        print(f"Insertion complete to {table_name}. Total records: {total_records}")


def main(language: str, version: str, output_file: str) -> None:
    con = duckdb.connect(output_file)
    con.execute("CREATE OR REPLACE TABLE pages (page_id VARCHAR, title VARCHAR)")
    con.execute("CREATE OR REPLACE TABLE wikidata (page_id VARCHAR, wikidata_id VARCHAR)")
    con.execute("CREATE OR REPLACE TABLE redirects (page_id VARCHAR, rd_to VARCHAR)")

    base_url = f"https://dumps.wikimedia.org/{language}wiki/{version}/"
    page_file = f"{base_url}{language}wiki-{version}-page.sql.gz"
    props_file = f"{base_url}{language}wiki-{version}-page_props.sql.gz"
    redirect_file = f"{base_url}{language}wiki-{version}-redirect.sql.gz"

    try:
        print("Processing pages...")
        page_pattern = re.compile(r"\((\d+),0,'(.+?)',")
        process_and_insert(con, page_file, "pages", page_pattern)

        print("\nProcessing page properties...")
        props_pattern = re.compile(r"\((\d+),'wikibase_item','(Q\d+)',")
        process_and_insert(con, props_file, "wikidata", props_pattern)

        print("\nProcessing redirects...")
        redirect_pattern = re.compile(r"\((\d+),0,'(.*?)',")
        process_and_insert(con, redirect_file, "redirects", redirect_pattern)

        print("\nCreating final wiki_map table...")
        con.execute("""
            CREATE OR REPLACE TABLE wiki_map AS
                SELECT
                        p.page_id,
                        p.title,
                        self_props.wikidata_id AS wikidata_id,
                        CASE WHEN r.page_id IS NOT NULL THEN true ELSE false END AS is_redirect,
                        p_target.page_id as redirect_to
                FROM pages p
                LEFT JOIN wikidata self_props ON p.page_id = self_props.page_id
                LEFT JOIN redirects r ON p.page_id = r.page_id

                LEFT JOIN pages p_target ON r.rd_to = p_target.title
                LEFT JOIN wikidata target_props ON p_target.page_id = target_props.page_id
        """)

        # Dropping intermediate tables
        con.execute("DROP TABLE pages")
        con.execute("DROP TABLE wikidata")
        con.execute("DROP TABLE redirects")

        print("Creating indexes...")
        con.execute("CREATE INDEX title ON wiki_map (title)")
        con.execute("CREATE INDEX page_id ON wiki_map (page_id)")

        row_count = con.execute("SELECT count(*) FROM wiki_map").fetchone()[0]
        print("\n！""===============================")
        print(f"Created database: {output_file}")
        print(f"Total entities: {row_count:,}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        con.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate wiki_map DuckDB database from Wikipedia dumps.")
    parser.add_argument('--language', '-l', type=str, default='en', help='Language code for Wikipedia dumps.')
    parser.add_argument('--version', '-v', type=str, default='latest', help='Version of Wikipedia dumps.')
    parser.add_argument('--output_file', '-o', type=str, default='wikipedia_to_wikidata.duckdb', help='Output DuckDB database file name.')

    args = parser.parse_args()
    main(args.language, args.version, args.output_file)
