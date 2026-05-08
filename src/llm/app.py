#!/usr/bin/python

# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from flask import Flask, request, jsonify, Response
import json
import time
import random
import re
import os
import logging

from openfeature import api
from openfeature.contrib.provider.flagd import FlagdProvider

from opentelemetry import trace, metrics
from opentelemetry.trace import Status, StatusCode
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry._logs import set_logger_provider

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# Initialize OpenTelemetry
service_name = os.environ.get('OTEL_SERVICE_NAME', 'llm')
tracer = trace.get_tracer_provider().get_tracer(service_name)
meter = metrics.get_meter_provider().get_meter(service_name)

# Initialize Logs
logger_provider = LoggerProvider(
    resource=Resource.create(
        {
            'service.name': service_name,
        }
    ),
)
set_logger_provider(logger_provider)
log_exporter = OTLPLogExporter(insecure=True)
logger_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))
handler = LoggingHandler(level=logging.NOTSET, logger_provider=logger_provider)

# Attach OTLP handler to logger
logger = logging.getLogger('llm')
logger.addHandler(handler)

product_review_summaries = None
product_review_summaries_file_path = "./product-review-summaries.json"

inaccurate_product_review_summaries = None
inaccurate_product_review_summaries_file_path = "./inaccurate-product-review-summaries.json"

def load_product_review_summaries(file_path):
    try:
        with open(file_path, 'r') as file:

            """
            Converts a JSON string into an internal dictionary optimized for quick lookups.
            The keys of the internal dictionary will be product_ids.
            """
            try:
                data = json.load(file)
                summaries = data.get("product-review-summaries", [])

                # Create a dictionary where product_id is the key
                # and the value is the summary
                product_review_summaries = {}
                for product in summaries:
                    product_id = product.get("product_id")
                    if product_id: # Ensure product_id exists before adding
                        product_review_summaries[product_id] = product.get("product_review_summary")
                return product_review_summaries
            except json.JSONDecodeError:
                print("Error: Invalid JSON string provided during initialization.")
                return {}

    except FileNotFoundError:
        app.logger.error(f"Error: The file '{product_review_summaries_file_path}' was not found.")
    except json.JSONDecodeError:
        app.logger.error(f"Error: Failed to decode JSON from the file '{product_review_summaries_file_path}'. Check for malformed JSON.")
    except Exception as e:
        app.logger.error(f"An unexpected error occurred: {e}")


def generate_response(product_id):

    """Generate a response by providing the pre-generated summary for the specified product"""
    
    with tracer.start_as_current_span("generate_response") as span:
        span.set_attribute("app.product.id", product_id)
        
        product_review_summary = None

        llm_inaccurate_response = check_feature_flag("llmInaccurateResponse")
        app.logger.info(f"llmInaccurateResponse feature flag: {llm_inaccurate_response}")
        if llm_inaccurate_response and product_id == "L9ECAV7KIM":
            app.logger.info(f"Returning an inaccurate response for product_id: {product_id}")
            product_review_summary = inaccurate_product_review_summaries.get(product_id)
            span.set_attribute("app.llm.response.accurate", False)
        else:
            product_review_summary = product_review_summaries.get(product_id)
            span.set_attribute("app.llm.response.accurate", True)

        app.logger.info(f"product_review_summary is: {product_review_summary}")
        
        if product_review_summary:
            span.set_attribute("app.llm.summary.length", len(product_review_summary))
            span.set_attribute("app.llm.summary.tokens", len(product_review_summary.split()))
        else:
            span.set_attribute("app.llm.summary.length", 0)
            span.set_attribute("app.llm.summary.tokens", 0)

        return product_review_summary

def parse_product_id(last_message):
    match = re.search(r"product ID:([A-Z0-9]+)", last_message)
    if match:
        return match.group(1).strip()

    match = re.search(r"product ID, but make the answer inaccurate:([A-Z0-9]+)", last_message)
    if match:
        return match.group(1).strip()

    raise ValueError("product ID not found in input message")

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    with tracer.start_as_current_span("chat_completions") as span:
        data = request.json
        messages = data.get('messages', [])
        stream = data.get('stream', False)
        model = data.get('model', 'astronomy-llm')
        tools = data.get('tools', None)

        app.logger.info(f"Received a chat completion request: '{messages}'")
        
        span.set_attribute("app.llm.model", model)
        span.set_attribute("app.llm.stream", stream)
        span.set_attribute("app.llm.messages.count", len(messages))

        last_message = messages[-1]["content"]

        app.logger.info(f"last_message is: '{last_message}'")

        if 'What age(s) is this recommended for?' in last_message:
            response_text = 'This product is recommended for ages 7 and above.'
            return build_response(model, messages, response_text, span)
        elif 'Were there any negative reviews?' in last_message:
            response_text = 'No, there were no reviews less than three stars for this product.'
            return build_response(model, messages, response_text, span)
        elif not ('Can you summarize the product reviews?' in last_message or 'Based on the tool results, answer the original question about product ID' in last_message):
            response_text = 'Sorry, I\'m not able to answer that question.'
            return build_response(model, messages, response_text, span)

        # otherwise, process the product review summary
        try:
            product_id = parse_product_id(last_message)
            span.set_attribute("app.product.id", product_id)
        except ValueError as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, description=str(e)))
            error_response = {
                "error": {
                    "message": "Unable to parse product ID from request",
                    "type": "invalid_request_error",
                    "param": "messages",
                    "code": None
                }
            }
            return jsonify(error_response), 400

        if tools is not None:

            tool_args = f"{{\"product_id\": \"{product_id}\"}}"

            app.logger.info(f"Processing a tool call with args: '{tool_args}'")

            app.logger.info(f"The model is: {model}")
            if model.endswith("rate-limit"):
                app.logger.info(f"Returning a rate limit error")
                span.set_attribute("app.llm.error.type", "rate_limit_exceeded")
                span.set_status(Status(StatusCode.ERROR, description="Rate limit exceeded"))
                response = {
                    "error": {
                        "message": "Rate limit reached. Please try again later.",
                        "type": "rate_limit_exceeded",
                        "param": "null",
                        "code": "null"
                    }
                }
                return jsonify(response), 429
            else:
                # Non-streaming response
                prompt_tokens = sum(len(m.get("content", "").split()) for m in messages)
                completion_tokens = len("requesting a tool call".split())
                
                response = {
                    "id": f"chatcmpl-mock-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "requesting a tool call",
                            "tool_calls": [{
                                "id": "call",
                                "type": "function",
                                "function": {
                                    "name": "fetch_product_reviews",
                                    "arguments": tool_args
                                }
                            }]
                        },
                        "finish_reason": "tool_calls"
                    }],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens
                    }
                }
                
                span.set_attribute("app.llm.tokens.prompt", prompt_tokens)
                span.set_attribute("app.llm.tokens.completion", completion_tokens)
                span.set_attribute("app.llm.tokens.total", prompt_tokens + completion_tokens)
                
                return jsonify(response)

        else:
            # Generate the response
            response_text = generate_response(product_id)

            return build_response(model, messages, response_text, span)

def build_response(model, messages, response_text, span=None):
    app.logger.info(f"Processing a response: '{response_text}'")

    prompt_tokens = sum(len(m.get("content", "").split()) for m in messages)
    completion_tokens = len(response_text.split()) if response_text else 0
    total_tokens = prompt_tokens + completion_tokens
    
    if span:
        span.set_attribute("app.llm.tokens.prompt", prompt_tokens)
        span.set_attribute("app.llm.tokens.completion", completion_tokens)
        span.set_attribute("app.llm.tokens.total", total_tokens)

    response = {
        "id": f"chatcmpl-mock-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response_text
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }
    }
    return jsonify(response)

@app.route('/v1/models', methods=['GET'])
def list_models():
    """List available models"""
    return jsonify({
        "object": "list",
        "data": [
            {
                "id": "astronomy-llm",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "astronomy-shop"
            }
        ]
    })

def check_feature_flag(flag_name: str):
    # Initialize OpenFeature
    client = api.get_client()
    return client.get_boolean_value(flag_name, False)

if __name__ == '__main__':

    api.set_provider(FlagdProvider(host=os.environ.get('FLAGD_HOST', 'flagd'), port=os.environ.get('FLAGD_PORT', 8013)))
    product_review_summaries = load_product_review_summaries(product_review_summaries_file_path)
    inaccurate_product_review_summaries = load_product_review_summaries(inaccurate_product_review_summaries_file_path)

    app.logger.info(product_review_summaries)

    print("OpenAI API server starting on http://localhost:8000")
    print("Set your OpenAI base URL to: http://localhost:8000/v1")
    app.run(host='0.0.0.0', port=8000, debug=True)
