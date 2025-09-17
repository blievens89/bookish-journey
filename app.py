import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import io
from urllib.parse import urljoin, urlparse, urldefrag
from urllib.robotparser import RobotFileParser
import time
from collections import deque
import threading

st.set_page_config(
    page_title="Thorough Domain Crawler", 
    page_icon="🕷️",
    layout="wide"
)

st.title("🕷️ Thorough Domain Crawler")
st.caption("Deep crawl domains to find actual page counts (not just sitemaps)")

# Thread-safe storage for crawl results
crawl_results = {}
crawl_lock = threading.Lock()

def clean_domain(url_or_domain):
    """Clean and standardise domain input"""
    if not url_or_domain:
        return None
    
    url_or_domain = url_or_domain.strip()
    
    # If it looks like a full URL, extract domain
    if url_or_domain.startswith(('http://', 'https://')):
        parsed = urlparse(url_or_domain)
        domain = parsed.netloc
        base_url = f"{parsed.scheme}://{parsed.netloc}"
    else:
        domain = url_or_domain
        domain = re.sub(r'^www\.', '', domain.lower())
        base_url = f"https://{domain}"
    
    return domain, base_url

def is_valid_page(url, domain):
    """Check if URL is a valid page to count"""
    parsed = urlparse(url)
    
    # Must be on same domain
    if domain not in parsed.netloc:
        return False
    
    # Skip common non-page files
    skip_extensions = {'.pdf', '.jpg', '.jpeg', '.png', '.gif', '.css', '.js', '.ico', 
                      '.xml', '.txt', '.zip', '.doc', '.docx', '.xls', '.xlsx'}
    
    path = parsed.path.lower()
    if any(path.endswith(ext) for ext in skip_extensions):
        return False
    
    # Skip common non-content paths
    skip_paths = {'/wp-admin/', '/admin/', '/login/', '/wp-content/', '/assets/', 
                 '/images/', '/css/', '/js/', '/fonts/'}
    
    if any(skip_path in path for skip_path in skip_paths):
        return False
    
    return True

def get_links_from_page(url, domain, session, max_retries=2):
    """Extract links from a single page"""
    links = set()
    
    for attempt in range(max_retries):
        try:
            response = session.get(url, timeout=15, allow_redirects=True)
            
            if response.status_code == 200 and 'text/html' in response.headers.get('content-type', ''):
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find all links
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    
                    # Convert relative URLs to absolute
                    absolute_url = urljoin(url, href)
                    
                    # Remove fragments (#section)
                    clean_url = urldefrag(absolute_url)[0]
                    
                    # Check if it's a valid page
                    if is_valid_page(clean_url, domain):
                        links.add(clean_url)
                
                break  # Success, no need to retry
                
        except Exception as e:
            if attempt == max_retries - 1:
                st.warning(f"Failed to crawl {url}: {str(e)[:100]}")
            time.sleep(1)  # Brief pause before retry
    
    return links

def crawl_domain_thorough(domain, max_pages=500, max_depth=3, timeout_minutes=10):
    """Thoroughly crawl a domain to count pages"""
    domain, base_url = clean_domain(domain)
    
    if not domain:
        return 0, "Invalid domain", []
    
    # Setup
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (compatible; ThoroughCrawler/1.0)'
    })
    
    # Check robots.txt
    try:
        rp = RobotFileParser()
        rp.set_url(f"{base_url}/robots.txt")
        rp.read()
        can_fetch = rp.can_fetch('*', base_url)
        if not can_fetch:
            return 0, "Robots.txt disallows crawling", []
    except:
        pass  # Continue if robots.txt check fails
    
    # Breadth-first search
    visited = set()
    queue = deque([(base_url, 0)])  # (url, depth)
    found_pages = []
    
    start_time = time.time()
    timeout_seconds = timeout_minutes * 60
    
    while queue and len(visited) < max_pages:
        # Check timeout
        if time.time() - start_time > timeout_seconds:
            return len(found_pages), f"Timeout after {timeout_minutes}min ({len(found_pages)} pages found)", found_pages
        
        current_url, depth = queue.popleft()
        
        # Skip if already visited or too deep
        if current_url in visited or depth > max_depth:
            continue
        
        visited.add(current_url)
        found_pages.append(current_url)
        
        # Update progress in UI
        with crawl_lock:
            crawl_results[domain] = {
                'pages_found': len(found_pages),
                'status': f"Crawling... ({len(found_pages)} pages, depth {depth})"
            }
        
        # Get links from this page if we haven't reached max depth
        if depth < max_depth:
            try:
                links = get_links_from_page(current_url, domain, session)
                
                # Add new links to queue
                for link in links:
                    if link not in visited and len(queue) + len(visited) < max_pages * 2:
                        queue.append((link, depth + 1))
                        
            except Exception as e:
                continue
    
    total_time = time.time() - start_time
    method = f"Deep crawl ({total_time:.1f}s, depth {max_depth})"
    
    return len(found_pages), method, found_pages

def crawl_multiple_domains(domains, max_pages, max_depth, timeout_minutes, progress_callback=None):
    """Crawl multiple domains with progress tracking"""
    results = []
    
    # Initialize results tracking
    with crawl_lock:
        for domain in domains:
            crawl_results[domain] = {'pages_found': 0, 'status': 'Waiting...'}
    
    def crawl_single_domain(domain):
        try:
            pages, method, urls = crawl_domain_thorough(domain, max_pages, max_depth, timeout_minutes)
            return {
                'Domain': domain,
                'Pages': pages,
                'Method': method,
                'Sample_URLs': urls[:10]  # First 10 URLs as sample
            }
        except Exception as e:
            return {
                'Domain': domain,
                'Pages': 0,
                'Method': f'Error: {str(e)[:100]}',
                'Sample_URLs': []
            }
    
    # Process domains sequentially to avoid overwhelming servers
    for i, domain in enumerate(domains):
        result = crawl_single_domain(domain)
        results.append(result)
        
        if progress_callback:
            progress_callback(i + 1, len(domains))
    
    return results

# Main interface
st.markdown("### Deep Domain Crawling")
st.markdown("⚠️ **Warning**: This performs actual crawling and is much slower than sitemap analysis")

col1, col2 = st.columns([2, 1])

with col1:
    url_input = st.text_area(
        label="URLs/Domains to Crawl",
        placeholder="https://example.com\nexample2.co.uk\nhttps://site3.org",
        height=150,
        label_visibility="collapsed"
    )

with col2:
    st.markdown("**Crawl Settings**")
    
    max_pages = st.number_input(
        "Max pages per domain",
        min_value=10,
        max_value=1000,
        value=100,
        step=10,
        help="Limit pages to avoid very long crawls"
    )
    
    max_depth = st.slider(
        "Max crawl depth",
        min_value=1,
        max_value=5,
        value=2,
        help="How many link levels to follow"
    )
    
    timeout_minutes = st.number_input(
        "Timeout per domain (minutes)",
        min_value=1,
        max_value=30,
        value=5,
        help="Maximum time to spend on each domain"
    )

# Input validation and preview
if url_input:
    lines = [line.strip() for line in url_input.split('\n') if line.strip()]
    domains_preview = []
    
    for line in lines:
        domain_info = clean_domain(line)
        if domain_info:
            domains_preview.append(domain_info[0])
    
    unique_domains = list(dict.fromkeys(domains_preview))
    
    if unique_domains:
        st.info(f"Ready to crawl {len(unique_domains)} domains")
        estimated_time = len(unique_domains) * timeout_minutes
        st.caption(f"⏱️ Estimated maximum time: {estimated_time} minutes")

# Crawl button and processing
if st.button("🕷️ Start Deep Crawl", type="primary") and url_input:
    lines = [line.strip() for line in url_input.split('\n') if line.strip()]
    domains_to_crawl = []
    
    for line in lines:
        domain_info = clean_domain(line)
        if domain_info:
            domains_to_crawl.append(domain_info[0])
    
    unique_domains = list(dict.fromkeys(domains_to_crawl))
    
    if unique_domains:
        # Progress tracking
        progress_bar = st.progress(0)
        status_container = st.container()
        
        def update_progress(completed, total):
            progress = completed / total
            progress_bar.progress(progress)
            
            # Show live status for each domain
            with status_container:
                cols = st.columns(min(3, len(unique_domains)))
                for i, domain in enumerate(unique_domains[:len(cols)]):
                    with cols[i % len(cols)]:
                        with crawl_lock:
                            if domain in crawl_results:
                                st.metric(
                                    domain[:20] + "..." if len(domain) > 20 else domain,
                                    crawl_results[domain]['pages_found'],
                                    crawl_results[domain]['status']
                                )
        
        start_time = time.time()
        
        # Process domains
        results = crawl_multiple_domains(
            unique_domains, 
            max_pages, 
            max_depth, 
            timeout_minutes, 
            update_progress
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Clear progress indicators
        progress_bar.empty()
        status_container.empty()
        
        if results:
            # Convert to DataFrame
            df = pd.DataFrame(results)
            
            # Display results
            st.success(f"Crawl complete! Processed {len(results)} domains in {processing_time:.1f} seconds")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            successful = len(df[df['Pages'] > 0])
            total_pages = df['Pages'].sum()
            avg_pages = df[df['Pages'] > 0]['Pages'].mean() if successful > 0 else 0
            max_pages_found = df['Pages'].max()
            
            with col1:
                st.metric("Domains Crawled", len(results))
            with col2:
                st.metric("Successful Crawls", successful)
            with col3:
                st.metric("Total Pages Found", f"{total_pages:,}")
            with col4:
                st.metric("Largest Site", f"{max_pages_found:,} pages")
            
            # Show results table
            st.markdown("### Crawl Results")
            
            # Sort by pages descending
            df_display = df[['Domain', 'Pages', 'Method']].sort_values('Pages', ascending=False)
            st.dataframe(df_display, use_container_width=True)
            
            # Sample URLs for verification
            if st.checkbox("Show sample URLs found"):
                for _, row in df.iterrows():
                    if row['Pages'] > 0 and row['Sample_URLs']:
                        with st.expander(f"{row['Domain']} - Sample URLs"):
                            for url in row['Sample_URLs']:
                                st.text(url)
            
            # Create Excel file for download
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Main results
                df_export = df[['Domain', 'Pages']].sort_values('Pages', ascending=False)
                df_export.to_excel(writer, sheet_name='Results', index=False)
                
                # Detailed results
                df[['Domain', 'Pages', 'Method']].sort_values('Pages', ascending=False).to_excel(
                    writer, sheet_name='Detailed', index=False
                )
            
            excel_data = output.getvalue()
            
            # Download button
            st.download_button(
                label="📥 Download Crawl Results (Excel)",
                data=excel_data,
                file_name=f"thorough_crawl_results_{int(time.time())}.xlsx",
                mime="application/vnd.openxlxml"
            )

# Sidebar info
with st.sidebar:
    st.markdown("### 🕷️ How Deep Crawling Works")
    st.markdown("""
    **This app actually crawls websites:**
    1. Starts at homepage
    2. Follows internal links
    3. Counts unique HTML pages
    4. Respects crawl limits
    
    **vs Sitemap Analysis:**
    - ✅ Finds pages not in sitemaps
    - ✅ More accurate for small sites
    - ❌ Much slower
    - ❌ Uses more resources
    """)
    
    st.markdown("### ⚙️ Settings Explained")
    st.markdown("""
    **Max Pages**: Safety limit per domain
    
    **Max Depth**: How deep to follow links
    - Depth 1: Homepage + direct links
    - Depth 2: + links from those pages
    - Depth 3+: Even deeper
    
    **Timeout**: Prevents infinite crawls
    """)
    
    st.markdown("### 🎯 Best For")
    st.markdown("""
    - Sites without sitemaps
    - Small to medium sites
    - Accurate page counting
    - When you have time to wait
    """)
    
    st.markdown("---")
    st.caption("⚠️ Be respectful: This tool makes many requests to websites")
