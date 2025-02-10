import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import fitz  # PyMuPDF
from typing import Dict, List, Optional
import os
import re
from collections import Counter
import json
from datetime import datetime
import pandas as pd

class PDFProcessor:
    """Handles multiple PDF files processing and text extraction with page tracking."""
    
    @staticmethod
    def extract_text_from_pdf(pdf_file) -> Dict[str, any]:
        """Extracts text and metadata from PDF with page numbers."""
        try:
            pdf_bytes = pdf_file.read()
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            # Dictionary to store text content with page numbers
            pages_content = {}
            full_text = ""
            tables = []
            images = []
            
            # Extract text and content page by page
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                page_text = page.get_text()
                
                # Extract tables with better error handling
                try:
                    blocks = page.get_text("blocks")
                    page_tables = []
                    for block in blocks:
                        if _is_table(block):
                            page_tables.append({
                                'page': page_num + 1,
                                'content': block[4] if isinstance(block, tuple) and len(block) > 4 else str(block)
                            })
                    tables.extend(page_tables)
                except Exception as e:
                    st.warning(f"Could not extract tables from page {page_num + 1}: {str(e)}")
                    continue
                
                # Extract images
                for img_index, img in enumerate(page.get_images(full=True)):
                    images.append({
                        'page': page_num + 1,
                        'index': img_index,
                        'type': img[1],
                    })
                
                pages_content[page_num + 1] = {
                    'text': page_text,
                    'tables': [t for t in tables if t['page'] == page_num + 1],
                    'images': [i for i in images if i['page'] == page_num + 1]
                }
                full_text += f"\n[Page {page_num + 1}]\n{page_text}"
            
            metadata = {
                "title": pdf_document.metadata.get("title", "Unknown"),
                "author": pdf_document.metadata.get("author", "Unknown"),
                "total_pages": len(pdf_document),
                "total_tables": len(tables),
                "total_images": len(images),
                "creation_date": pdf_document.metadata.get("creationDate", "Unknown"),
                "modification_date": pdf_document.metadata.get("modDate", "Unknown")
            }
            
            return {
                "text": full_text,
                "pages_content": pages_content,
                "metadata": metadata,
                "tables": tables,
                "images": images
            }
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return {"text": "", "pages_content": {}, "metadata": {}, "tables": [], "images": []}

def _is_table(block):
    """Helper function to identify if a text block might be a table."""
    try:
        # PyMuPDF blocks are tuples with (x0, y0, x1, y1, "text", block_no, block_type)
        if isinstance(block, tuple) and len(block) >= 7:
            text = block[4]  # Get the text content
            if not isinstance(text, str):
                return False
                
            lines = text.split('\n')
            if len(lines) <= 2:
                return False
                
            # Check for regular structure in lines
            word_counts = [len(line.split()) for line in lines if line.strip()]
            if not word_counts:
                return False
                
            # Check if most lines have similar number of elements
            avg_count = sum(word_counts) / len(word_counts)
            deviation = sum(abs(c - avg_count) for c in word_counts) / len(word_counts)
            
            return deviation < 2 and avg_count > 2
            
    except (AttributeError, IndexError, TypeError):
        return False
        
    return False

class ConversationManager:
    """Manages conversation history and memory."""
    
    def __init__(self):
        self.memory = ConversationBufferMemory(return_messages=True)
        self.conversation_logs = []
    
    def add_interaction(self, user_input: str, assistant_response: str, references: List[Dict]):
        """Add an interaction to the conversation history."""
        timestamp = datetime.now().isoformat()
        interaction = {
            'timestamp': timestamp,
            'user_input': user_input,
            'assistant_response': assistant_response,
            'references': references
        }
        self.conversation_logs.append(interaction)
        
        # Update memory
        self.memory.save_context(
            {"input": user_input},
            {"output": assistant_response}
        )
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the recent conversation history."""
        messages = self.memory.chat_memory.messages[-5:]  # Last 5 messages
        summary = []
        for msg in messages:
            summary.append(f"{msg.type}: {msg.content[:100]}...")
        return "\n".join(summary)
    
    def export_conversation(self, format: str = 'json') -> str:
        """Export conversation history in specified format."""
        if format == 'json':
            return json.dumps(self.conversation_logs, indent=2)
        elif format == 'csv':
            df = pd.DataFrame(self.conversation_logs)
            return df.to_csv(index=False)
        return ""

class MultiPDFChat:
    """Enhanced chat interactions with multiple PDFs."""
    
    def __init__(self):
        self.setup_models()
        self.pdf_processor = PDFProcessor()
        self.conversation_manager = ConversationManager()
        self.initialize_session_state()
        
    def setup_models(self):
        """Initialize available models."""
        self.models = {
            "Deepseek-1.5B": ChatOllama(
                model="deepseek-r1:1.5b",
                temperature=0.3,
                base_url="http://localhost:11434"
            ),
            "Deepseek-8B": ChatOllama(
                model="deepseek-r1:8b",
                temperature=0.3,
                base_url="http://localhost:11434"
            ),
            "LLaMA3-8B": ChatOllama(
                model="llama3:8b",
                temperature=0.3,
                base_url="http://localhost:11434"
            )
        }
        
        # Initialize memory separately for each model
        self.memories = {
            name: ConversationBufferMemory(return_messages=True)
            for name in self.models.keys()
        }
    
    def initialize_session_state(self):
        """Initialize enhanced session state variables."""
        if "pdfs_data" not in st.session_state:
            st.session_state.pdfs_data = {}
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "selected_model" not in st.session_state:
            st.session_state.selected_model = "Deepseek-8B"
        if "conversation_mode" not in st.session_state:
            st.session_state.conversation_mode = "chat"  # or 'analysis'
    
    def create_system_prompt(self) -> str:
        """Creates enhanced system prompt with document context."""
        # Get document contexts with sections
        doc_contexts = []
        for pdf_name, data in st.session_state.pdfs_data.items():
            # Get document sections if available
            sections_info = ""
            if "sections" in data["metadata"] and data["metadata"]["sections"]:
                sections = data["metadata"]["sections"]
                sections_info = "\nDocument sections:\n" + "\n".join(
                    f"- {s['title']} (Pages {s['start_page']}-{s['end_page']})"
                    for s in sections
                )
            
            doc_contexts.append(
                f"Document: {pdf_name}\n"
                f"Title: {data['metadata']['title']}\n"
                f"Author: {data['metadata']['author']}\n"
                f"Pages: {data['metadata']['total_pages']}"
                f"{sections_info}\n\n"
                f"Content preview:\n{data['text'][:1000]}"
            )

        separator = "-" * 40
        documents_text = "\n\n".join(doc_contexts)
        
        system_prompt = (
            "You are an advanced research assistant analyzing multiple PDF documents. "
            "Your goal is to provide detailed, accurate information while citing specific sources.\n\n"
            f"Documents available:\n{separator}\n{documents_text}\n{separator}\n\n"
            "Instructions:\n"
            "1. ALWAYS cite sources using [Document: name, Page: X] format\n"
            "2. When information spans multiple pages, cite all relevant pages\n"
            "3. If you find tables or figures, mention them specifically\n"
            "4. If content is from a specific section, mention the section name\n"
            "5. If you're unsure about anything, ask for clarification\n"
            "6. If you can't find relevant information, say so clearly\n\n"
            "Provide clear, concise answers focused on the document content. Stay factual and cite your sources."
        )
        
        return system_prompt
    
    def find_page_references(self, response: str, prompt: str) -> list[dict]:
        """
        Extracts document and page references from the response text.
        Looks for patterns like [Document: name, Page: X] or similar variations.
        
        Args:
            response (str): The assistant's response text
            prompt (str): The user's original prompt
            
        Returns:
            list[dict]: List of reference dictionaries with 'document' and 'page' keys
        """
        references = []
        
        # Handle case where response is not a string
        if not isinstance(response, str):
            return references
            
        # Regular expression patterns to match different reference formats
        patterns = [
            # [Document: doc_name, Page: X]
            r'\[Document:\s*([^,\]]+),\s*Page:\s*(\d+)\]',
            # [doc_name, Page X]
            r'\[([^,\]]+),\s*Page\s*(\d+)\]',
            # [doc_name, p. X]
            r'\[([^,\]]+),\s*p\.\s*(\d+)\]'
        ]
        
        found_references = set()  # Use set to avoid duplicates
        
        for pattern in patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            for match in matches:
                doc_name = match.group(1).strip()
                page_num = int(match.group(2))
                
                # Validate that the document exists in the session state
                if doc_name in st.session_state.pdfs_data:
                    # Validate page number is within document's range
                    total_pages = st.session_state.pdfs_data[doc_name]["metadata"]["total_pages"]
                    if 1 <= page_num <= total_pages:
                        # Create unique string for deduplication
                        ref_key = f"{doc_name}:{page_num}"
                        if ref_key not in found_references:
                            references.append({
                                "document": doc_name,
                                "page": page_num
                            })
                            found_references.add(ref_key)
        
        # Sort references by document name and page number
        references.sort(key=lambda x: (x["document"], x["page"]))
        
        return references
    
    def run(self):
        """Enhanced main application interface."""
        st.title("📚 Advanced Multi-PDF Chat Assistant")
        st.caption("Intelligent PDF Analysis and Chat with Memory")
        
        # Enhanced sidebar
        with st.sidebar:
            st.header("🔧 Settings")
            
            # Model selection
            selected_model = st.selectbox(
                "Choose Model",
                list(self.models.keys()),
                index=list(self.models.keys()).index(st.session_state.selected_model)
            )
            st.session_state.selected_model = selected_model
            
            # Conversation mode
            st.session_state.conversation_mode = st.radio(
                "Mode",
                ["Chat", "Document Analysis", "Summary"],
                format_func=lambda x: f"📝 {x}"
            )
            
            # Export options
            if st.button("Export Conversation"):
                format = st.radio("Export Format", ["JSON", "CSV"])
                data = self.conversation_manager.export_conversation(format.lower())
                st.download_button(
                    "Download",
                    data,
                    file_name=f"conversation_export.{format.lower()}",
                    mime=f"text/{format.lower()}"
                )
        
        # Main content area
        # PDF upload section
        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=['pdf'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            for pdf_file in uploaded_files:
                if pdf_file.name not in st.session_state.pdfs_data:
                    with st.spinner(f"Processing {pdf_file.name}..."):
                        pdf_data = self.pdf_processor.extract_text_from_pdf(pdf_file)
                        st.session_state.pdfs_data[pdf_file.name] = pdf_data
        
        # Document overview in a horizontal layout
        if st.session_state.pdfs_data:
            st.subheader("📑 Loaded Documents")
            doc_cols = st.columns(min(3, len(st.session_state.pdfs_data)))
            for idx, (pdf_name, pdf_data) in enumerate(st.session_state.pdfs_data.items()):
                with doc_cols[idx % 3]:
                    with st.expander(f"📄 {pdf_name}", expanded=False):
                        st.json(pdf_data["metadata"])
                        st.write(f"📊 Tables found: {len(pdf_data['tables'])}")
                        st.write(f"🖼️ Images found: {len(pdf_data['images'])}")
        
        # Add a visual separator
        st.divider()
        
        # Main interface area
        if st.session_state.conversation_mode == "Chat":
            self._display_chat_interface()
        elif st.session_state.conversation_mode == "Document Analysis":
            self._display_analysis_interface()
        else:  # Summary mode
            self._display_summary_interface()
    
    def _display_chat_interface(self):
        """Display the chat interface with memory."""
        st.subheader(f"💬 Chat")
        
        # Display chat history with references
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "references" in message and message["references"]:
                    with st.expander("📚 Sources"):
                        for ref in message["references"]:
                            st.markdown(f"- {ref['document']}, Page {ref['page']}")
        
        # Chat input
        if prompt := st.chat_input("Ask about the documents..."):
            user_message = {"role": "user", "content": prompt}
            st.session_state.chat_history.append(user_message)
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            try:
                # Create the chat chain
                messages = [
                    ("system", self.create_system_prompt()),
                    ("human", f"Based on the provided documents, {prompt}")
                ]
                
                # Add context from memory
                memory = self.memories[st.session_state.selected_model]
                if memory.chat_memory.messages:
                    messages.extend([
                        (msg.type, msg.content)
                        for msg in memory.chat_memory.messages[-4:]  # Last 4 messages
                    ])
                
                # Get response
                chain = ChatPromptTemplate.from_messages(messages) | self.models[st.session_state.selected_model]
                response = chain.invoke({})
                
                # Extract the actual response content
                response_content = ""
                if hasattr(response, 'content'):
                    response_content = response.content
                elif isinstance(response, str):
                    response_content = response
                elif isinstance(response, dict) and 'content' in response:
                    response_content = response['content']
                else:
                    response_content = str(response)
                
                # Clean up the response content
                # Remove any think tags if present
                response_content = re.sub(r'<think>.*?</think>', '', response_content, flags=re.DOTALL)
                response_content = response_content.strip()
                
                # Extract references
                references = self.find_page_references(response_content, prompt)
                
                # Display response
                with st.chat_message("assistant"):
                    st.markdown(response_content)
                    if references:
                        with st.expander("📚 Sources"):
                            for ref in references:
                                st.markdown(f"- {ref['document']}, Page {ref['page']}")
                
                # Update conversation history
                assistant_message = {
                    "role": "assistant",
                    "content": response_content,
                    "references": references
                }
                st.session_state.chat_history.append(assistant_message)
                
                # Update conversation manager
                self.conversation_manager.add_interaction(prompt, response_content, references)
            
            except Exception as e:
                st.error(f"Error: {str(e)}")
                # Log the full error for debugging
                print(f"Full error: {e}")
    
    def _display_analysis_interface(self):
        """Display document analysis interface."""
        st.subheader("📊 Document Analysis")
        
        if st.session_state.pdfs_data:
            selected_doc = st.selectbox(
                "Select Document",
                list(st.session_state.pdfs_data.keys())
            )
            
            if selected_doc:
                doc_data = st.session_state.pdfs_data[selected_doc]
                
                # Document statistics
                st.write("### 📈 Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Pages", doc_data["metadata"]["total_pages"])
                with col2:
                    st.metric("Tables", doc_data["metadata"]["total_tables"])
                with col3:
                    st.metric("Images", doc_data["metadata"]["total_images"])
                
                # Section/Page Range Selection
                st.write("### 📖 Content Navigation")
                
                # Check if document has sections
                has_sections = "sections" in doc_data["metadata"] and doc_data["metadata"]["sections"]
                
                if has_sections:
                    # Section selection
                    sections = doc_data["metadata"]["sections"]
                    section_titles = [section["title"] for section in sections]
                    selected_section = st.selectbox(
                        "Select Section",
                        section_titles,
                        key="section_selector"
                    )
                    
                    # Find selected section details
                    section = next(s for s in sections if s["title"] == selected_section)
                    start_page = section["start_page"]
                    end_page = section["end_page"]
                    
                    st.info(f"Section spans pages {start_page} to {end_page}")
                
                # Page range selection
                st.write("#### Page Range Selection")
                col1, col2 = st.columns(2)
                with col1:
                    start_page = st.number_input(
                        "Start Page",
                        min_value=1,
                        max_value=doc_data["metadata"]["total_pages"],
                        value=1
                    )
                with col2:
                    end_page = st.number_input(
                        "End Page",
                        min_value=start_page,
                        max_value=doc_data["metadata"]["total_pages"],
                        value=min(start_page + 5, doc_data["metadata"]["total_pages"])
                    )
                
                # Display content for selected range
                if start_page <= end_page:
                    st.write(f"### 📝 Content (Pages {start_page} to {end_page})")
                    for page_num in range(start_page, end_page + 1):
                        with st.expander(f"Page {page_num}"):
                            page_content = doc_data["pages_content"][page_num]
                            
                            # Show section if available
                            if "section" in page_content and page_content["section"]:
                                st.info(f"Section: {page_content['section']}")
                            
                            # Show page content
                            st.text_area(
                                "Content",
                                page_content["text"],
                                height=200,
                                key=f"page_{page_num}"
                            )
                            
                            # Show tables if available
                            if page_content["tables"]:
                                st.write("#### Tables on this page")
                                for idx, table in enumerate(page_content["tables"]):
                                    st.code(table["content"], language="text")
                            
                            # Show image placeholders if available
                            if page_content["images"]:
                                st.write(f"#### Images on this page: {len(page_content['images'])}")
    
    def _display_summary_interface(self):
        """Display document summary interface."""
        st.subheader("📋 Document Summary")
        
        if st.session_state.pdfs_data:
            # Overall statistics
            total_pages = sum(data["metadata"]["total_pages"] for data in st.session_state.pdfs_data.values())
            total_tables = sum(data["metadata"]["total_tables"] for data in st.session_state.pdfs_data.values())
            total_images = sum(data["metadata"]["total_images"] for data in st.session_state.pdfs_data.values())
            
            st.write("### 📊 Overall Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Documents", len(st.session_state.pdfs_data))
            with col2:
                st.metric("Total Pages", total_pages)
            with col3:
                st.metric("Total Elements", total_tables + total_images)
            
            # Recent conversation summary
            st.write("### 💭 Recent Conversation")
            st.code(self.conversation_manager.get_conversation_summary())


if __name__ == "__main__":
    chat_app = MultiPDFChat()
    chat_app.run()