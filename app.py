from transformers import pipeline
import gradio as gr
from newspaper import Article

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text, max_length=130, min_length=30):
    if len(text.strip().split()) < 50:
        return "Input text is too short to summarize."
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

def summarize_url(url, max_length=130, min_length=30):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return summarize_text(article.text, max_length, min_length)
    except Exception as e:
        return f"Error: {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown("# 📰 News & Article Summarizer")

    with gr.Tab("Paste Text"):
        input_text = gr.Textbox(lines=10, label="Paste your article text here")
        output_summary = gr.Textbox(label="Summary")
        summarize_button = gr.Button("Summarize")
        summarize_button.click(summarize_text, inputs=input_text, outputs=output_summary)

    with gr.Tab("From URL"):
        input_url = gr.Textbox(label="Enter article URL")
        output_url_summary = gr.Textbox(label="Summary")
        summarize_url_button = gr.Button("Summarize URL")
        summarize_url_button.click(summarize_url, inputs=input_url, outputs=output_url_summary)

demo.launch()
