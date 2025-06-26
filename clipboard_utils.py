# clipboard_utils.py - Complete working implementation
import streamlit as st
import streamlit.components.v1 as components
import uuid
from typing import Optional

def create_copy_button(text: str, button_text: str = "📋 Copy", 
                      success_message: str = "✅ Copied!", 
                      key: Optional[str] = None) -> bool:
    """
    Create a working copy button that actually copies to clipboard
    
    Args:
        text: Text to copy to clipboard
        button_text: Text/emoji for the button
        success_message: Message to show on successful copy
        key: Unique key for the button
    
    Returns:
        bool: True if copy button was clicked
    """
    if key is None:
        key = f"copy_btn_{str(uuid.uuid4())[:8]}"
    
    # Create unique IDs for this copy button
    button_id = f"btn_{key}"
    text_id = f"text_{key}"
    
    # Clean text for JavaScript (escape quotes and newlines)
    clean_text = text.replace('\\', '\\\\').replace("'", "\\'").replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r')
    
    # HTML with JavaScript for reliable clipboard copying
    copy_button_html = f"""
    <div style="margin: 5px 0;">
        <button 
            id="{button_id}"
            onclick="copyToClipboard_{key}()" 
            style="
                background-color: #ff4b4b;
                color: white;
                border: none;
                padding: 8px 12px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 14px;
                font-family: 'Source Sans Pro', sans-serif;
                transition: background-color 0.2s;
            "
            onmouseover="this.style.backgroundColor='#ff6b6b'"
            onmouseout="this.style.backgroundColor='#ff4b4b'"
        >
            {button_text}
        </button>
        <span id="status_{key}" style="margin-left: 10px; color: green; font-weight: bold;"></span>
    </div>
    
    <textarea 
        id="{text_id}" 
        style="
            position: absolute; 
            left: -9999px; 
            opacity: 0;
            height: 1px;
            width: 1px;
        "
        readonly
    >{text}</textarea>
    
    <script>
        function copyToClipboard_{key}() {{
            const textArea = document.getElementById('{text_id}');
            const button = document.getElementById('{button_id}');
            const status = document.getElementById('status_{key}');
            
            // Method 1: Modern Clipboard API (preferred)
            if (navigator.clipboard && window.isSecureContext) {{
                navigator.clipboard.writeText(`{clean_text}`).then(function() {{
                    status.innerHTML = '{success_message}';
                    button.style.backgroundColor = '#28a745';
                    setTimeout(() => {{ 
                        status.innerHTML = ''; 
                        button.style.backgroundColor = '#ff4b4b';
                    }}, 2000);
                }}).catch(function(err) {{
                    console.error('Clipboard API failed:', err);
                    fallbackCopyTextToClipboard_{key}();
                }});
            }} else {{
                // Method 2: Fallback for older browsers
                fallbackCopyTextToClipboard_{key}();
            }}
        }}
        
        function fallbackCopyTextToClipboard_{key}() {{
            const textArea = document.getElementById('{text_id}');
            const status = document.getElementById('status_{key}');
            const button = document.getElementById('{button_id}');
            
            textArea.style.position = 'fixed';
            textArea.style.left = '0';
            textArea.style.top = '0';
            textArea.style.opacity = '1';
            textArea.style.width = '2em';
            textArea.style.height = '2em';
            textArea.focus();
            textArea.select();
            
            try {{
                const successful = document.execCommand('copy');
                if (successful) {{
                    status.innerHTML = '{success_message}';
                    button.style.backgroundColor = '#28a745';
                    setTimeout(() => {{ 
                        status.innerHTML = ''; 
                        button.style.backgroundColor = '#ff4b4b';
                    }}, 2000);
                }} else {{
                    status.innerHTML = '❌ Copy failed';
                    setTimeout(() => {{ status.innerHTML = ''; }}, 2000);
                }}
            }} catch (err) {{
                console.error('Fallback copy failed:', err);
                status.innerHTML = '❌ Copy not supported';
                setTimeout(() => {{ status.innerHTML = ''; }}, 2000);
            }}
            
            textArea.style.position = 'absolute';
            textArea.style.left = '-9999px';
            textArea.style.opacity = '0';
            textArea.style.width = '1px';
            textArea.style.height = '1px';
        }}
    </script>
    """
    
    # Render the copy button
    components.html(copy_button_html, height=60)
    
    return False  # We handle the copy in JavaScript

def create_copy_text_area(text: str, label: str = "Response", 
                         height: int = 150, key: Optional[str] = None) -> None:
    """
    Create a text area with built-in copy functionality
    
    Args:
        text: Text to display and allow copying
        label: Label for the text area
        height: Height of the text area
        key: Unique key for the widget
    """
    if key is None:
        key = f"copy_area_{str(uuid.uuid4())[:8]}"
    
    # Create text area with select-all helper text
    st.text_area(
        label=f"{label} (Ctrl+A to select all, then Ctrl+C to copy)",
        value=text,
        height=height,
        key=f"textarea_{key}",
        help="Select all text (Ctrl+A or Cmd+A) then copy (Ctrl+C or Cmd+C)"
    )

def create_downloadable_text(text: str, filename: str, 
                           button_text: str = "💾 Download") -> None:
    """
    Create a download button for text content
    
    Args:
        text: Text content to download
        filename: Suggested filename
        button_text: Text for the download button
    """
    st.download_button(
        label=button_text,
        data=text,
        file_name=filename,
        mime="text/plain",
        help=f"Download as {filename}"
    )

def create_copy_section(text: str, title: str = "Copy Response", 
                       show_download: bool = True, key: Optional[str] = None) -> None:
    """
    Create a complete copy section with multiple copy options
    
    Args:
        text: Text to copy
        title: Section title
        show_download: Whether to show download option
        key: Unique key for the section
    """
    if key is None:
        key = f"copy_section_{str(uuid.uuid4())[:8]}"
    
    with st.expander(title, expanded=False):
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            # Method 1: Enhanced copy button
            create_copy_button(
                text=text, 
                button_text="📋 Copy to Clipboard",
                key=f"enhanced_{key}"
            )
        
        with col2:
            if show_download:
                # Method 2: Download option
                create_downloadable_text(
                    text=text,
                    filename=f"ai_response_{key}.txt",
                    button_text="💾 Download"
                )
        
        with col3:
            # Method 3: Show character count
            st.metric("Characters", len(text))
        
        # Method 4: Selectable text area
        create_copy_text_area(
            text=text,
            label="Select and copy manually",
            height=100,
            key=f"area_{key}"
        )

def inject_copy_script():
    """Inject copy functionality into Streamlit app"""
    copy_script = """
    <script>
    // Global copy function for simple usage
    function copyToClipboard(text, buttonId) {
        if (navigator.clipboard && window.isSecureContext) {
            navigator.clipboard.writeText(text).then(function() {
                if (buttonId) {
                    const button = document.getElementById(buttonId);
                    if (button) {
                        const originalText = button.innerHTML;
                        button.innerHTML = '✅ Copied!';
                        button.style.backgroundColor = '#28a745';
                        setTimeout(() => {
                            button.innerHTML = originalText;
                            button.style.backgroundColor = '#ff4b4b';
                        }, 2000);
                    }
                }
            }).catch(function(err) {
                console.error('Could not copy text: ', err);
                alert('Copy failed. Please select and copy manually.');
            });
        } else {
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = text;
            textArea.style.position = 'fixed';
            textArea.style.left = '-999999px';
            textArea.style.top = '-999999px';
            document.body.appendChild(textArea);
            textArea.focus();
            textArea.select();
            try {
                document.execCommand('copy');
                if (buttonId) {
                    const button = document.getElementById(buttonId);
                    if (button) {
                        const originalText = button.innerHTML;
                        button.innerHTML = '✅ Copied!';
                        button.style.backgroundColor = '#28a745';
                        setTimeout(() => {
                            button.innerHTML = originalText;
                            button.style.backgroundColor = '#ff4b4b';
                        }, 2000);
                    }
                }
            } catch (err) {
                console.error('Fallback copy failed: ', err);
                alert('Copy failed. Please select and copy manually.');
            }
            document.body.removeChild(textArea);
        }
    }
    
    // Auto-select text areas on focus
    document.addEventListener('DOMContentLoaded', function() {
        const textAreas = document.querySelectorAll('textarea');
        textAreas.forEach(function(textArea) {
            textArea.addEventListener('focus', function() {
                this.select();
            });
        });
    });
    </script>
    """
    
    components.html(copy_script, height=0)

# Simple usage function for quick integration
def simple_copy_button(text: str, label: str = "Copy") -> None:
    """Simple copy button for quick usage"""
    create_copy_button(text, button_text=f"📋 {label}")