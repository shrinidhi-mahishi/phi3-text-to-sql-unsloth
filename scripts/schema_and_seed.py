# Demo schema used in prompts (mirrors a small analytics warehouse)
SCHEMA = """
TABLE products (
  sku TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  family TEXT NOT NULL
);
TABLE price_list (
  sku TEXT NOT NULL,
  tier TEXT NOT NULL,              -- Enterprise | SMB
  unit_price_usd REAL NOT NULL,
  currency TEXT NOT NULL,
  effective_from TEXT NOT NULL,
  PRIMARY KEY (sku, tier)
);
TABLE accounts (
  account_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  segment TEXT NOT NULL,           -- Enterprise | SMB
  region TEXT NOT NULL,
  arr_usd REAL NOT NULL,
  champion TEXT
);
TABLE entitlements (
  account_id TEXT NOT NULL,
  sku TEXT NOT NULL,
  seats INTEGER NOT NULL,
  discount_pct REAL NOT NULL,
  PRIMARY KEY (account_id, sku)
);
""".strip()

# (instruction, sql) pairs — expand freely; keep SQL dialect = SQLite-ish ANSI
EXAMPLES: list[tuple[str, str]] = [
    (
        "List all product names and SKUs.",
        "SELECT sku, name FROM products ORDER BY name;",
    ),
    (
        "What is the Enterprise list price for Cloud Shield (CS-ENT)?",
        "SELECT unit_price_usd FROM price_list WHERE sku = 'CS-ENT' AND tier = 'Enterprise';",
    ),
    (
        "Show SMB and Enterprise prices for Data Fabric.",
        "SELECT tier, unit_price_usd FROM price_list WHERE sku = 'DF-ENT' ORDER BY tier;",
    ),
    (
        "Which accounts are in the EMEA region?",
        "SELECT account_id, name, champion FROM accounts WHERE region = 'EMEA';",
    ),
    (
        "Find the champion for Northstar Manufacturing.",
        "SELECT champion FROM accounts WHERE name = 'Northstar Manufacturing';",
    ),
    (
        "Get entitlements for account ACC-NS including discount percent.",
        "SELECT sku, seats, discount_pct FROM entitlements WHERE account_id = 'ACC-NS';",
    ),
    (
        "Compute net unit price for Northstar on CS-ENT using list price and entitlement discount.",
        """SELECT p.unit_price_usd * (1 - e.discount_pct / 100.0) AS net_unit_price
FROM entitlements e
JOIN accounts a ON a.account_id = e.account_id
JOIN price_list p ON p.sku = e.sku AND p.tier = a.segment
WHERE a.name = 'Northstar Manufacturing' AND e.sku = 'CS-ENT';""",
    ),
    (
        "List products in the Security family.",
        "SELECT sku, name FROM products WHERE family = 'Security';",
    ),
    (
        "How many seats does Helios Energy have for OBS-SU?",
        """SELECT e.seats
FROM entitlements e
JOIN accounts a ON a.account_id = e.account_id
WHERE a.name = 'Helios Energy' AND e.sku = 'OBS-SU';""",
    ),
    (
        "Show all Enterprise-tier prices above 100 USD.",
        "SELECT sku, unit_price_usd FROM price_list WHERE tier = 'Enterprise' AND unit_price_usd > 100 ORDER BY unit_price_usd DESC;",
    ),
    (
        "Which SMB accounts are in APAC?",
        "SELECT account_id, name, arr_usd FROM accounts WHERE segment = 'SMB' AND region = 'APAC';",
    ),
    (
        "Return average ARR by region.",
        "SELECT region, AVG(arr_usd) AS avg_arr FROM accounts GROUP BY region ORDER BY avg_arr DESC;",
    ),
    (
        "Find SKUs with both SMB and Enterprise price rows.",
        "SELECT sku FROM price_list GROUP BY sku HAVING COUNT(DISTINCT tier) = 2;",
    ),
    (
        "List account names with discount greater than 10 percent on any SKU.",
        """SELECT DISTINCT a.name, e.sku, e.discount_pct
FROM accounts a
JOIN entitlements e ON a.account_id = e.account_id
WHERE e.discount_pct > 10
ORDER BY e.discount_pct DESC;""",
    ),
    (
        "What is the effective_from date for Observability Suite Enterprise pricing?",
        "SELECT effective_from FROM price_list WHERE sku = 'OBS-SU' AND tier = 'Enterprise';",
    ),
    (
        "Count products per family.",
        "SELECT family, COUNT(*) AS product_count FROM products GROUP BY family;",
    ),
    (
        "Show Orbit Retail's segment and region.",
        "SELECT segment, region FROM accounts WHERE name = 'Orbit Retail';",
    ),
    (
        "Join products to price_list and show name, tier, and unit price.",
        """SELECT pr.name, pl.tier, pl.unit_price_usd
FROM products pr
JOIN price_list pl ON pr.sku = pl.sku
ORDER BY pr.name, pl.tier;""",
    ),
    (
        "Which account has the highest ARR?",
        "SELECT name, arr_usd FROM accounts ORDER BY arr_usd DESC LIMIT 1;",
    ),
    (
        "List entitlements where seats are at least 200.",
        "SELECT account_id, sku, seats FROM entitlements WHERE seats >= 200 ORDER BY seats DESC;",
    ),
    (
        "Get Cloud Shield SMB list price.",
        "SELECT unit_price_usd FROM price_list WHERE sku = 'CS-ENT' AND tier = 'SMB';",
    ),
    (
        "Find accounts missing any entitlement rows.",
        """SELECT a.account_id, a.name
FROM accounts a
LEFT JOIN entitlements e ON a.account_id = e.account_id
WHERE e.account_id IS NULL;""",
    ),
    (
        "Total seats entitled across all accounts for CS-ENT.",
        "SELECT SUM(seats) AS total_seats FROM entitlements WHERE sku = 'CS-ENT';",
    ),
    (
        "Show champions for Enterprise segment accounts in North America.",
        "SELECT name, champion FROM accounts WHERE segment = 'Enterprise' AND region = 'North America';",
    ),
    (
        "Return Data Fabric Enterprise unit price and currency.",
        "SELECT unit_price_usd, currency FROM price_list WHERE sku = 'DF-ENT' AND tier = 'Enterprise';",
    ),
    (
        "List SKU and discount for Northstar Manufacturing ordered by discount descending.",
        """SELECT e.sku, e.discount_pct
FROM entitlements e
JOIN accounts a ON a.account_id = e.account_id
WHERE a.name = 'Northstar Manufacturing'
ORDER BY e.discount_pct DESC;""",
    ),
    (
        "How many price rows exist per tier?",
        "SELECT tier, COUNT(*) AS n FROM price_list GROUP BY tier;",
    ),
    (
        "Find products whose name contains 'Suite'.",
        "SELECT sku, name FROM products WHERE name LIKE '%Suite%';",
    ),
    (
        "Select account_id and name for ARR between 100000 and 600000.",
        "SELECT account_id, name, arr_usd FROM accounts WHERE arr_usd BETWEEN 100000 AND 600000 ORDER BY arr_usd;",
    ),
    (
        "Get the minimum Enterprise unit price across all SKUs.",
        "SELECT MIN(unit_price_usd) AS min_enterprise_price FROM price_list WHERE tier = 'Enterprise';",
    ),
    (
        "Show Helios Energy account_id and ARR.",
        "SELECT account_id, arr_usd FROM accounts WHERE name = 'Helios Energy';",
    ),
    (
        "List distinct regions in the accounts table.",
        "SELECT DISTINCT region FROM accounts ORDER BY region;",
    ),
    (
        "For each account, count how many SKUs they are entitled to.",
        """SELECT a.name, COUNT(e.sku) AS sku_count
FROM accounts a
LEFT JOIN entitlements e ON a.account_id = e.account_id
GROUP BY a.name
ORDER BY sku_count DESC;""",
    ),
    (
        "Return Observability Suite SMB price.",
        "SELECT unit_price_usd FROM price_list WHERE sku = 'OBS-SU' AND tier = 'SMB';",
    ),
    (
        "Find entitlements with discount_pct equal to 18.",
        "SELECT account_id, sku, seats FROM entitlements WHERE discount_pct = 18;",
    ),
    (
        "Show product family for SKU DF-ENT.",
        "SELECT family FROM products WHERE sku = 'DF-ENT';",
    ),
    (
        "List Enterprise prices effective from 2024-01-01.",
        "SELECT sku, unit_price_usd FROM price_list WHERE tier = 'Enterprise' AND effective_from = '2024-01-01';",
    ),
    (
        "Get name and champion for ACC-OR.",
        "SELECT name, champion FROM accounts WHERE account_id = 'ACC-OR';",
    ),
    (
        "Calculate Northstar's net price for DF-ENT.",
        """SELECT p.unit_price_usd * (1 - e.discount_pct / 100.0) AS net_unit_price
FROM entitlements e
JOIN accounts a ON a.account_id = e.account_id
JOIN price_list p ON p.sku = e.sku AND p.tier = a.segment
WHERE a.name = 'Northstar Manufacturing' AND e.sku = 'DF-ENT';""",
    ),
    (
        "Which SKUs are in the Data family?",
        "SELECT sku, name FROM products WHERE family = 'Data';",
    ),
]
