"""
Chat-with-Documents: A Streamlit app for querying documents with traditional IR techniques,
sentiment analysis, and document summarization
"""

import streamlit as st
import os
import re
import io
import pandas as pd
import nltk
import pickle
import tempfile
import time
import math
from collections import Counter, defaultdict
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.sentiment import SentimentIntensityAnalyzer
import PyPDF2

# Download necessary NLTK resources
def download_nltk_resources():
    resources = ['punkt', 'stopwords', 'vader_lexicon']
    for resource in resources:
        try:
            if resource == 'punkt':
                nltk.data.find('tokenizers/punkt')
            elif resource == 'stopwords':
                nltk.data.find('corpora/stopwords')
            elif resource == 'vader_lexicon':
                nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download(resource, quiet=True)

# Download resources at startup
download_nltk_resources()

class SentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of text and return scores and classification"""
        scores = self.analyzer.polarity_scores(text)
        
        # Classify based on compound score
        if scores['compound'] >= 0.05:
            sentiment = 'positive'
            color = 'green'
        elif scores['compound'] <= -0.05:
            sentiment = 'negative'
            color = 'red'
        else:
            sentiment = 'neutral'
            color = 'orange'
            
        return {
            'sentiment': sentiment,
            'color': color,
            'scores': scores,
            'compound': scores['compound']
        }
    
    def get_sentiment_emoji(self, sentiment):
        """Return emoji based on sentiment"""
        emoji_map = {
            'positive': '😊',
            'negative': '😞',
            'neutral': '😐'
        }
        return emoji_map.get(sentiment, '😐')

class DocumentSummarizer:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
    
    def extract_key_sentences(self, text, num_sentences=3):
        """Extract key sentences using TF-IDF scoring"""
        sentences = sent_tokenize(text)
        if len(sentences) <= num_sentences:
            return sentences
        
        # Calculate word frequencies
        words = word_tokenize(text.lower())
        words = [self.stemmer.stem(word) for word in words if word.isalnum() and word not in self.stop_words]
        word_freq = Counter(words)
        
        # Score sentences
        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            sentence_words = word_tokenize(sentence.lower())
            sentence_words = [self.stemmer.stem(word) for word in sentence_words if word.isalnum() and word not in self.stop_words]
            
            score = 0
            for word in sentence_words:
                if word in word_freq:
                    score += word_freq[word]
            
            if len(sentence_words) > 0:
                sentence_scores[i] = score / len(sentence_words)
        
        # Get top sentences
        top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:num_sentences]
        top_sentences = sorted([idx for idx, _ in top_sentences])  # Sort by original order
        
        return [sentences[i] for i in top_sentences]
    
    def summarize_document(self, text, max_sentences=5):
        """Create a summary of the document"""
        key_sentences = self.extract_key_sentences(text, max_sentences)
        return ' '.join(key_sentences)

class DocumentProcessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.sentiment_analyzer = SentimentAnalyzer()
        self.summarizer = DocumentSummarizer()
        
    def preprocess_text(self, text):
        """Clean and preprocess text for indexing"""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stopwords and apply stemming
        tokens = [self.stemmer.stem(token) for token in tokens if token not in self.stop_words]
        # Join back to string
        return ' '.join(tokens)
    
    def chunk_text(self, text, chunk_size=200, overlap=50):
        """Split text into overlapping chunks"""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            tokens = word_tokenize(sentence)
            
            if current_length + len(tokens) > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                # Keep last few sentences for overlap
                overlap_tokens = current_chunk[-overlap:] if overlap < len(current_chunk) else current_chunk
                current_chunk = overlap_tokens
                current_length = len(overlap_tokens)
            
            current_chunk.append(sentence)
            current_length += len(tokens)
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks
    
    def extract_text_from_pdf(self, pdf_file):
        """Extract text content from a PDF file"""
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    
    def process_document(self, file, file_type):
        """Process a document based on its type"""
        if file_type == 'text/plain':
            text = file.read().decode('utf-8')
            # Reset file pointer for potential reuse
            file.seek(0)
        elif file_type == 'application/pdf':
            text = self.extract_text_from_pdf(file)
            # Reset file pointer for potential reuse
            file.seek(0)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        # Generate document summary and sentiment
        summary = self.summarizer.summarize_document(text)
        sentiment_analysis = self.sentiment_analyzer.analyze_sentiment(text)
        
        chunks = self.chunk_text(text)
        processed_chunks = [self.preprocess_text(chunk) for chunk in chunks]
        
        return chunks, processed_chunks, summary, sentiment_analysis, text

class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(list)
        self.document_lengths = {}
        self.avg_doc_length = 0
        self.total_docs = 0
        self.original_chunks = []
        self.chunk_sentiments = []
        self.document_summaries = []
        self.document_sentiments = []
        self.document_texts = []
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.sentiment_analyzer = SentimentAnalyzer()
        
    def add_document(self, doc_id, processed_text, original_text, summary=None, sentiment=None, full_text=None):
        """Add a document to the index"""
        # Store original text and metadata
        self.original_chunks.append(original_text)
        
        # Analyze chunk sentiment
        chunk_sentiment = self.sentiment_analyzer.analyze_sentiment(original_text)
        self.chunk_sentiments.append(chunk_sentiment)
        
        # Store document-level data (only for the first chunk of each document)
        if summary and sentiment and full_text:
            self.document_summaries.append(summary)
            self.document_sentiments.append(sentiment)
            self.document_texts.append(full_text)
        
        # Update document stats
        tokens = processed_text.split()
        self.document_lengths[doc_id] = len(tokens)
        self.total_docs += 1
        
        # Build term frequency dictionary for this document
        term_freq = Counter(tokens)
        
        # Update the inverted index
        for term, freq in term_freq.items():
            self.index[term].append((doc_id, freq))
            
        # Update average document length
        self.avg_doc_length = sum(self.document_lengths.values()) / self.total_docs
            
    def search_tfidf(self, query, top_k=5):
        """Search documents using TF-IDF scoring"""
        # Preprocess query
        query = query.lower()
        query = re.sub(r'[^\w\s]', ' ', query)
        query_tokens = word_tokenize(query)
        query_tokens = [self.stemmer.stem(token) for token in query_tokens if token not in self.stop_words]
        
        # Calculate document scores
        scores = defaultdict(float)
        for token in query_tokens:
            if token in self.index:
                # Calculate IDF
                idf = math.log((self.total_docs + 1) / (len(self.index[token]) + 1)) + 1
                
                # Score each document containing this term
                for doc_id, term_freq in self.index[token]:
                    # TF-IDF score
                    tf = term_freq / self.document_lengths[doc_id]
                    scores[doc_id] += tf * idf
        
        # Get top_k results
        top_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for doc_id, score in top_results:
            if score > 0:  # Only include documents with matching terms
                highlight_text = self.highlight_matches(self.original_chunks[doc_id], query_tokens)
                sentiment = self.chunk_sentiments[doc_id] if doc_id < len(self.chunk_sentiments) else None
                results.append({
                    'chunk_id': doc_id,
                    'text': self.original_chunks[doc_id],
                    'highlighted_text': highlight_text,
                    'score': score,
                    'sentiment': sentiment
                })
                
        return results
    
    def search_bm25(self, query, top_k=5, k1=1.5, b=0.75):
        """Search documents using BM25 scoring"""
        # Preprocess query
        query = query.lower()
        query = re.sub(r'[^\w\s]', ' ', query)
        query_tokens = word_tokenize(query)
        query_tokens = [self.stemmer.stem(token) for token in query_tokens if token not in self.stop_words]
        
        # Calculate document scores
        scores = defaultdict(float)
        for token in query_tokens:
            if token in self.index:
                # Calculate IDF component for BM25
                idf = math.log((self.total_docs - len(self.index[token]) + 0.5) / 
                              (len(self.index[token]) + 0.5) + 1)
                
                # Score each document containing this term
                for doc_id, term_freq in self.index[token]:
                    # BM25 score for this term and document
                    doc_length = self.document_lengths[doc_id]
                    numerator = term_freq * (k1 + 1)
                    denominator = term_freq + k1 * (1 - b + b * doc_length / self.avg_doc_length)
                    scores[doc_id] += idf * (numerator / denominator)
        
        # Get top_k results
        top_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for doc_id, score in top_results:
            if score > 0:  # Only include documents with matching terms
                highlight_text = self.highlight_matches(self.original_chunks[doc_id], query_tokens)
                sentiment = self.chunk_sentiments[doc_id] if doc_id < len(self.chunk_sentiments) else None
                results.append({
                    'chunk_id': doc_id,
                    'text': self.original_chunks[doc_id],
                    'highlighted_text': highlight_text,
                    'score': score,
                    'sentiment': sentiment
                })
                
        return results
    
    def highlight_matches(self, text, query_tokens):
        """Highlight query terms in the text for better interpretability"""
        highlighted = text
        for token in query_tokens:
            # Try to find word variations (handling stemming)
            pattern = r'\b\w*' + re.escape(token[:4]) + r'\w*\b'
            matches = re.finditer(pattern, text.lower())
            
            # Replace matches with highlighted version
            for match in matches:
                start, end = match.span()
                word = text[start:end]
                highlighted = highlighted.replace(word, f"**{word}**")
        
        return highlighted
    
    def get_sentiment_distribution(self):
        """Get distribution of sentiments across chunks"""
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        for sentiment_data in self.chunk_sentiments:
            sentiment_counts[sentiment_data['sentiment']] += 1
        return sentiment_counts
    
    def save(self, filename):
        """Save the index to a file"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'index': self.index,
                'document_lengths': self.document_lengths,
                'avg_doc_length': self.avg_doc_length,
                'total_docs': self.total_docs,
                'original_chunks': self.original_chunks,
                'chunk_sentiments': self.chunk_sentiments,
                'document_summaries': self.document_summaries,
                'document_sentiments': self.document_sentiments,
                'document_texts': self.document_texts
            }, f)
    
    def load(self, filename):
        """Load the index from a file"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.index = data['index']
            self.document_lengths = data['document_lengths']
            self.avg_doc_length = data['avg_doc_length']
            self.total_docs = data['total_docs']
            self.original_chunks = data['original_chunks']
            self.chunk_sentiments = data.get('chunk_sentiments', [])
            self.document_summaries = data.get('document_summaries', [])
            self.document_sentiments = data.get('document_sentiments', [])
            self.document_texts = data.get('document_texts', [])

class EvaluationMetrics:
    @staticmethod
    def precision_at_k(results, relevant_docs, k=5):
        """Calculate precision@k"""
        if not results or k <= 0:
            return 0.0
            
        # Get the top k results
        top_k = results[:k]
        
        # Count relevant documents in top k
        relevant_in_top_k = sum(1 for result in top_k if result['chunk_id'] in relevant_docs)
        
        return relevant_in_top_k / min(k, len(top_k)) if len(top_k) > 0 else 0.0
    
    @staticmethod
    def recall_at_k(results, relevant_docs, k=5):
        """Calculate recall@k"""
        if not results or not relevant_docs or k <= 0:
            return 0.0
            
        # Get the top k results
        top_k = results[:k]
        
        # Count relevant documents in top k
        relevant_in_top_k = sum(1 for result in top_k if result['chunk_id'] in relevant_docs)
        
        return relevant_in_top_k / len(relevant_docs) if len(relevant_docs) > 0 else 0.0

# Create a sample document if needed (for testing)
def create_sample_document():
    sample_text = """# Introduction to Information Retrieval

Information retrieval (IR) is the science of searching for information in documents, searching for documents themselves, searching for metadata that describe documents, or searching within databases, whether relational standalone databases or hypertext networked databases such as the World Wide Web. This field has revolutionized how we access and process information in the digital age.

## Traditional IR Methods

Traditional information retrieval systems use various algorithms to retrieve relevant documents. The most common methods include powerful techniques that have stood the test of time.

### Boolean Retrieval
The simplest form of retrieval using logical operators (AND, OR, NOT) to combine terms in queries. While basic, it provides precise control over search results.

### Vector Space Model
Documents and queries are represented as vectors in a multidimensional space, where each dimension corresponds to a term. Similarity is measured using metrics like cosine similarity. This approach is elegant and mathematically sound.

### TF-IDF Weighting
Term Frequency-Inverse Document Frequency (TF-IDF) is a numerical statistic that reflects how important a word is to a document in a collection. It increases proportionally to the number of times a word appears in the document but is offset by the frequency of the word in the collection. This method is highly effective for ranking documents.

### BM25 Ranking
Best Match 25 is a ranking function used by search engines to rank matching documents according to their relevance to a given search query. It is a probabilistic retrieval framework that has proven superior to many alternatives.

## Evaluation Metrics

Information retrieval systems are typically evaluated using metrics like precision, recall, F-measure, Mean Average Precision (MAP), and Normalized Discounted Cumulative Gain (NDCG). These metrics help measure the usefulness of documents based on their position in result lists.

## Modern Approaches

While traditional methods are still valuable, modern IR often incorporates machine learning and neural networks for improved performance. Learning to Rank (LTR) uses machine learning to improve ranking performance, while neural IR models use deep learning for document representation and matching. Transformer models use attention mechanisms for better understanding of language semantics.

However, traditional approaches remain relevant due to their interpretability, computational efficiency, and effectiveness for many use cases. The future of information retrieval lies in combining the best of both worlds - traditional methods and modern AI techniques."""
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.txt', mode='w') as f:
        f.write(sample_text)
        temp_filename = f.name
    
    return temp_filename

def display_sentiment_badge(sentiment_data):
    """Display a colored sentiment badge"""
    sentiment = sentiment_data['sentiment']
    color = sentiment_data['color']
    emoji = SentimentAnalyzer().get_sentiment_emoji(sentiment)
    compound = sentiment_data['compound']
    
    # Create colored HTML badge
    badge_html = f"""
    <div style="display: inline-block; padding: 4px 8px; border-radius: 12px; 
                background-color: {color}; color: white; font-size: 12px; font-weight: bold;">
        {emoji} {sentiment.upper()} ({compound:.2f})
    </div>
    """
    return badge_html

def main():
    st.set_page_config(page_title="Enhanced Chat with Documents", page_icon="📚", layout="wide")
    
    st.title("📚 Enhanced Chat with Documents")
    st.write("Upload documents and ask questions using traditional IR techniques with sentiment analysis and summarization")
    
    # Initialize session state variables
    if 'processor' not in st.session_state:
        st.session_state.processor = DocumentProcessor()
    if 'index' not in st.session_state:
        st.session_state.index = InvertedIndex()
    if 'documents_loaded' not in st.session_state:
        st.session_state.documents_loaded = False
    if 'search_method' not in st.session_state:
        st.session_state.search_method = "BM25"
    if 'evaluation_mode' not in st.session_state:
        st.session_state.evaluation_mode = False
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    if 'top_k' not in st.session_state:
        st.session_state.top_k = 5
    if 'show_sentiment' not in st.session_state:
        st.session_state.show_sentiment = True
    if 'show_summaries' not in st.session_state:
        st.session_state.show_summaries = True
    
    # Sidebar for document upload and settings
    with st.sidebar:
        st.header("Settings")
        
        # Document upload
        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader("Choose documents", accept_multiple_files=True, type=['txt', 'pdf'])
        
        # Create and use sample document option
        use_sample = st.checkbox("Use sample document for testing")
        if use_sample and st.button("Load Sample Document"):
            try:
                # Download required resources to make sure everything works
                download_nltk_resources()
                
                sample_file = create_sample_document()
                with open(sample_file, "rb") as f:
                    # Create a BytesIO object
                    file_like_object = io.BytesIO(f.read())
                    # Reset the pointer to the beginning
                    file_like_object.seek(0)
                    
                    # Create file object similar to what st.file_uploader returns
                    class FileObject:
                        def __init__(self, name, content):
                            self.name = name
                            self.content = content
                            
                        def read(self):
                            return self.content.read()
                            
                        def seek(self, pos):
                            return self.content.seek(pos)
                    
                    file_obj = FileObject('sample_document.txt', file_like_object)
                    
                    # Process document with sentiment and summary
                    chunks, processed_chunks, summary, sentiment, full_text = st.session_state.processor.process_document(
                        file_obj, 'text/plain')
                    
                    # Reset index
                    st.session_state.index = InvertedIndex()
                    
                    # Add first chunk with document metadata
                    doc_id = 0
                    st.session_state.index.add_document(doc_id, processed_chunks[0], chunks[0], summary, sentiment, full_text)
                    
                    # Add remaining chunks
                    for j in range(1, len(chunks)):
                        doc_id = j
                        st.session_state.index.add_document(doc_id, processed_chunks[j], chunks[j])
                    
                    st.session_state.documents_loaded = True
                    st.success(f"Loaded sample document with {len(st.session_state.index.original_chunks)} chunks!")
                
                # Clean up the temporary file
                os.unlink(sample_file)
                
            except Exception as e:
                st.error(f"Error loading sample document: {str(e)}")
        
        if uploaded_files and st.button("Process Documents"):
            try:
                # Download required resources to make sure everything works
                download_nltk_resources()
                
                # Reset index if new documents are uploaded
                st.session_state.index = InvertedIndex()
                
                progress_bar = st.progress(0)
                for i, file in enumerate(uploaded_files):
                    # Determine file type
                    if file.name.endswith('.pdf'):
                        file_type = 'application/pdf'
                    else:
                        file_type = 'text/plain'
                    
                    # Process document with sentiment and summary
                    chunks, processed_chunks, summary, sentiment, full_text = st.session_state.processor.process_document(file, file_type)
                    
                    # Add first chunk with document metadata
                    doc_id = len(st.session_state.index.original_chunks)
                    st.session_state.index.add_document(doc_id, processed_chunks[0], chunks[0], summary, sentiment, full_text)
                    
                    # Add remaining chunks
                    for j in range(1, len(chunks)):
                        doc_id = len(st.session_state.index.original_chunks)
                        st.session_state.index.add_document(doc_id, processed_chunks[j], chunks[j])
                    
                    # Update progress
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                st.session_state.documents_loaded = True
                st.success(f"Processed {len(uploaded_files)} documents into {len(st.session_state.index.original_chunks)} chunks!")
            
            except Exception as e:
                st.error(f"Error processing documents: {str(e)}")
        
        # Search method selection
        st.subheader("Search Method")
        st.session_state.search_method = st.radio(
            "Select retrieval algorithm",
            options=["TF-IDF", "BM25"],
            index=1
        )
        
        # Display options
        st.subheader("Display Options")
        st.session_state.show_sentiment = st.checkbox("Show sentiment analysis", value=True)
        st.session_state.show_summaries = st.checkbox("Show document summaries", value=True)
        
        # Advanced settings collapsible
        with st.expander("Advanced Settings"):
            st.session_state.top_k = st.slider("Number of results to show", min_value=1, max_value=20, value=5)
            
            # Evaluation mode toggle
            st.session_state.evaluation_mode = st.checkbox("Enable evaluation mode")
            
            # Save/Load index
            if st.session_state.documents_loaded:
                if st.button("Save Index"):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
                        st.session_state.index.save(tmp.name)
                        with open(tmp.name, "rb") as f:
                            st.download_button(
                                label="Download Index",
                                data=f.read(),
                                file_name="document_index.pkl",
                                mime="application/octet-stream"
                            )
                        # Clean up after download button is used
                        os.unlink(tmp.name)
            
            uploaded_index = st.file_uploader("Load saved index", type=['pkl'])
            if uploaded_index:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
                    tmp.write(uploaded_index.getvalue())
                    tmp_path = tmp.name
                
                if st.button("Load Index"):
                    try:
                        st.session_state.index.load(tmp_path)
                        st.session_state.documents_loaded = True
                        st.success(f"Loaded index with {len(st.session_state.index.original_chunks)} chunks!")
                    except Exception as e:
                        st.error(f"Error loading index: {e}")
                    finally:
                        # Clean up the temporary file
                        os.unlink(tmp_path)
    
    # Main content area
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Document summaries section
        if st.session_state.documents_loaded and st.session_state.show_summaries and st.session_state.index.document_summaries:
            st.subheader("📄 Document Summaries")
            for i, (summary, sentiment) in enumerate(zip(st.session_state.index.document_summaries, st.session_state.index.document_sentiments)):
                with st.expander(f"Document {i+1} Summary"):
                    st.write(summary)
                    if st.session_state.show_sentiment:
                        st.markdown(display_sentiment_badge(sentiment), unsafe_allow_html=True)
        
        # Query input
        st.subheader("🔍 Search Documents")
        query = st.text_input("Ask a question about your documents", key="query")
        col1_1, col1_2 = st.columns(2)
        
        with col1_1:
            search_button = st.button("Search", type="primary", disabled=not st.session_state.documents_loaded)
        
        with col1_2:
            clear_button = st.button("Clear Results")
            
        # Handle search
        if search_button and query:
            with st.spinner("Searching..."):
                try:
                    start_time = time.time()
                    
                    # Perform search based on selected method
                    if st.session_state.search_method == "TF-IDF":
                        results = st.session_state.index.search_tfidf(query, top_k=st.session_state.top_k)
                    else:  # BM25
                        results = st.session_state.index.search_bm25(query, top_k=st.session_state.top_k)
                    
                    search_time = time.time() - start_time
                    
                    # Store query and results in history
                    st.session_state.query_history.append({
                        "query": query,
                        "results": results,
                        "method": st.session_state.search_method,
                        "time": search_time
                    })
                except Exception as e:
                    st.error(f"Error performing search: {str(e)}")
        
        # Clear results
        if clear_button:
            st.session_state.query_history = []
        
        # Display search results
        if st.session_state.query_history:
            last_search = st.session_state.query_history[-1]
            
            st.subheader(f"Results for: '{last_search['query']}'")
            st.write(f"Found {len(last_search['results'])} results using {last_search['method']} in {last_search['time']:.4f} seconds")
            
            for i, result in enumerate(last_search['results']):
                with st.expander(f"Result {i+1} (Score: {result['score']:.4f})"):
                    # Display sentiment badge if available and enabled
                    if st.session_state.show_sentiment and result.get('sentiment'):
                        st.markdown(display_sentiment_badge(result['sentiment']), unsafe_allow_html=True)
                        st.markdown("---")
                    
                    st.markdown(result['highlighted_text'])
                    
                    # Only show evaluation options in evaluation mode
                    if st.session_state.evaluation_mode:
                        relevance = st.selectbox(
                            "Is this result relevant?",
                            options=["Not selected", "Relevant", "Not relevant"],
                            key=f"relevance_{len(st.session_state.query_history)-1}_{i}"
                        )
    
    with col2:
        # Metrics and visualizations
        if st.session_state.documents_loaded:
            st.subheader("📊 Document Statistics")
            
            # Basic stats
            st.write(f"Total chunks: {st.session_state.index.total_docs}")
            st.write(f"Average chunk length: {st.session_state.index.avg_doc_length:.1f} tokens")
            st.write(f"Vocabulary size: {len(st.session_state.index.index)}")
            st.write(f"Documents processed: {len(st.session_state.index.document_summaries)}")
            
            # Sentiment analysis overview
            if st.session_state.show_sentiment and st.session_state.index.chunk_sentiments:
                st.subheader("😊 Sentiment Analysis")
                
                # Get sentiment distribution
                sentiment_dist = st.session_state.index.get_sentiment_distribution()
                
                # Create sentiment distribution chart
                sentiment_df = pd.DataFrame(list(sentiment_dist.items()), columns=['Sentiment', 'Count'])
                
                # Color mapping for the chart
                color_map = {'positive': '#28a745', 'neutral': '#ffc107', 'negative': '#dc3545'}
                
                # Display sentiment metrics
                total_chunks = sum(sentiment_dist.values())
                if total_chunks > 0:
                    st.write("**Sentiment Distribution:**")
                    col_pos, col_neu, col_neg = st.columns(3)
                    
                    with col_pos:
                        pos_pct = (sentiment_dist['positive'] / total_chunks) * 100
                        st.metric("😊 Positive", f"{sentiment_dist['positive']}", f"{pos_pct:.1f}%")
                    
                    with col_neu:
                        neu_pct = (sentiment_dist['neutral'] / total_chunks) * 100
                        st.metric("😐 Neutral", f"{sentiment_dist['neutral']}", f"{neu_pct:.1f}%")
                    
                    with col_neg:
                        neg_pct = (sentiment_dist['negative'] / total_chunks) * 100
                        st.metric("😞 Negative", f"{sentiment_dist['negative']}", f"{neg_pct:.1f}%")
                
                # Sentiment distribution chart
                if not sentiment_df.empty:
                    st.bar_chart(sentiment_df.set_index('Sentiment'))
                
                # Overall document sentiment (if available)
                if st.session_state.index.document_sentiments:
                    st.write("**Document-Level Sentiment:**")
                    for i, doc_sentiment in enumerate(st.session_state.index.document_sentiments):
                        st.markdown(f"Document {i+1}: {display_sentiment_badge(doc_sentiment)}", unsafe_allow_html=True)
            
            # Evaluation metrics if in evaluation mode
            if st.session_state.evaluation_mode and st.session_state.query_history:
                st.subheader("📈 Evaluation")
                
                # Check if relevance judgments are available
                last_search = st.session_state.query_history[-1]
                relevant_docs = []
                
                # Get relevant document IDs from user feedback
                for i, result in enumerate(last_search['results']):
                    key = f"relevance_{len(st.session_state.query_history)-1}_{i}"
                    if key in st.session_state and st.session_state[key] == "Relevant":
                        relevant_docs.append(result['chunk_id'])
                
                if relevant_docs:
                    precision = EvaluationMetrics.precision_at_k(last_search['results'], relevant_docs, k=st.session_state.top_k)
                    recall = EvaluationMetrics.recall_at_k(last_search['results'], relevant_docs, k=st.session_state.top_k)
                    
                    # Display metrics
                    st.write(f"Precision@{st.session_state.top_k}: {precision:.2f}")
                    st.write(f"Recall@{st.session_state.top_k}: {recall:.2f}")
                    
                    # F1 score calculation
                    if precision + recall > 0:
                        f1 = 2 * precision * recall / (precision + recall)
                        st.write(f"F1 Score: {f1:.2f}")
                    
                    # Simple visualization of metrics
                    metrics_df = pd.DataFrame()
                    if precision + recall > 0:
                        metrics_df = pd.DataFrame({
                            'Metric': ['Precision', 'Recall', 'F1'],
                            'Value': [precision, recall, f1]
                        })
                    else:
                        metrics_df = pd.DataFrame({
                            'Metric': ['Precision', 'Recall'],
                            'Value': [precision, recall]
                        })
                    
                    st.bar_chart(metrics_df.set_index('Metric'))
                else:
                    st.write("Mark some results as relevant to see evaluation metrics.")
            
            # Advanced Analytics
            if st.session_state.index.chunk_sentiments:
                with st.expander("🔍 Advanced Analytics"):
                    # Sentiment score distribution
                    compound_scores = [s['compound'] for s in st.session_state.index.chunk_sentiments]
                    if compound_scores:
                        st.write("**Sentiment Score Distribution:**")
                        sentiment_score_df = pd.DataFrame({'Compound Score': compound_scores})
                        st.histogram_chart(sentiment_score_df)
                    
                    # Most positive and negative chunks
                    if len(st.session_state.index.chunk_sentiments) > 0:
                        # Find most positive chunk
                        most_positive_idx = max(range(len(st.session_state.index.chunk_sentiments)), 
                                              key=lambda i: st.session_state.index.chunk_sentiments[i]['compound'])
                        
                        # Find most negative chunk
                        most_negative_idx = min(range(len(st.session_state.index.chunk_sentiments)), 
                                              key=lambda i: st.session_state.index.chunk_sentiments[i]['compound'])
                        
                        st.write("**Most Positive Chunk:**")
                        pos_sentiment = st.session_state.index.chunk_sentiments[most_positive_idx]
                        st.markdown(display_sentiment_badge(pos_sentiment), unsafe_allow_html=True)
                        st.write(st.session_state.index.original_chunks[most_positive_idx][:200] + "...")
                        
                        st.write("**Most Negative Chunk:**")
                        neg_sentiment = st.session_state.index.chunk_sentiments[most_negative_idx]
                        st.markdown(display_sentiment_badge(neg_sentiment), unsafe_allow_html=True)
                        st.write(st.session_state.index.original_chunks[most_negative_idx][:200] + "...")
            
            # Search history
            if st.session_state.query_history:
                st.subheader("📝 Search History")
                
                for i, item in enumerate(reversed(st.session_state.query_history[-5:])):
                    query_sentiment = "neutral"
                    if item['results']:
                        # Calculate average sentiment of results
                        sentiment_scores = []
                        for result in item['results']:
                            if result.get('sentiment') and result['sentiment'].get('compound'):
                                sentiment_scores.append(result['sentiment']['compound'])
                        
                        if sentiment_scores:
                            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                            if avg_sentiment >= 0.05:
                                query_sentiment = "😊"
                            elif avg_sentiment <= -0.05:
                                query_sentiment = "😞"
                            else:
                                query_sentiment = "😐"
                    
                    st.write(f"{len(st.session_state.query_history) - i}. '{item['query']}' ({item['method']}) {query_sentiment}")
        
        # Help section
        with st.expander("❓ Help & Features"):
            st.markdown("""
            **New Features:**
            - 🎭 **Sentiment Analysis**: Each chunk and document gets sentiment scores (positive/neutral/negative)
            - 📝 **Document Summarization**: Automatic summaries of uploaded documents
            - 🎨 **Color-coded Results**: Green for positive, yellow for neutral, red for negative sentiment
            - 📊 **Enhanced Analytics**: Sentiment distribution charts and advanced metrics
            
            **How to Use:**
            1. Upload documents (TXT or PDF) or use the sample document
            2. Click "Process Documents" to index with sentiment analysis
            3. Enter search queries to find relevant content
            4. View results with sentiment indicators and highlighting
            5. Check document summaries in the main panel
            6. Explore sentiment analytics in the sidebar
            
            **Search Methods:**
            - **TF-IDF**: Classic term frequency-inverse document frequency
            - **BM25**: Modern probabilistic ranking (recommended)
            
            **Evaluation Mode:**
            - Enable to mark results as relevant/irrelevant
            - Get precision, recall, and F1 scores
            """)

if __name__ == "__main__":
    main()