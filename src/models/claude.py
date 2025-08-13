# src/models/claude.py
import os
import json
import boto3
from .base_model import SimplificationModel
from botocore.config import Config
from botocore.exceptions import ClientError


class ClaudeModel(SimplificationModel):
    """
    A wrapper for Anthropic's Claude models via Amazon Bedrock.
    """
    def __init__(self, api_model_name: str, temperature: float = 1.0):
        """
        Initializes the Claude model wrapper.
        
        Args:
            api_model_name: The exact model ID for the Bedrock API call
                            (e.g., "anthropic.claude-3-7-sonnet-20240715-v1:0").
            temperature: The sampling temperature to use.
        """
        try:
            # --- START: FINAL MODIFICATION ---

            # The profile name we created in `aws configure sso`
            profile_name = "claude-profile"
            
            # The region where the Bedrock model lives
            aws_region = "ap-southeast-2"

            # Create a session using the specified profile
            session = boto3.Session(profile_name=profile_name)

            # Configure a robust retry strategy
            retry_config = Config(
                retries={
                    'max_attempts': 10,
                    'mode': 'adaptive'
                }
            )

            # Create the Bedrock client from our session, specifying the service region and retry config
            self.bedrock_client = session.client(
                service_name='bedrock-runtime',
                region_name=aws_region,
                config=retry_config
            )
            
            # --- END: FINAL MODIFICATION ---

        except Exception as e:
            # A more specific error message now
            raise Exception(f"Failed to initialize Bedrock client for profile '{profile_name}'. Is the profile configured correctly?") from e

        self.model_name = api_model_name
        self.temperature = temperature
        print(f"✅ ClaudeModel initialized using profile '{profile_name}' for model: {self.model_name}")


    def simplify(self, prompt: str) -> tuple[str, str]:
        """
        Sends the prompt to the specified Claude model via Bedrock and returns the response.

        Returns:
            A tuple containing (raw_model_output, final_simplified_text).
        """
        try:
            # Construct the payload for the Bedrock API
            # Note: Claude's API structure is different from OpenAI's
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4096,
                "temperature": self.temperature,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}]
                    }
                ]
            })

            # Invoke the model
            response = self.bedrock_client.invoke_model(
                body=body, 
                modelId=self.model_name,
                accept='application/json',
                contentType='application/json'
            )

            # Parse the response
            response_body = json.loads(response.get('body').read())
            raw_output = response_body.get('content')[0].get('text').strip()

            # --- Parse for Chain-of-Thought ---
            cot_marker = "Simplified Document:"
            if cot_marker in raw_output:
                parts = raw_output.rsplit(cot_marker, 1)
                final_answer = parts[1].strip()
                return raw_output, final_answer
            else:
                return raw_output, raw_output

        except ClientError as e:
            # Check if the error is specifically a throttling exception
            if e.response['Error']['Code'] == 'ThrottlingException':
                print(f"❌ ThrottlingException detected in model call. Reporting to main loop.")
                raise # Re-raise the exception so the main loop can catch it
            else:
                # For any other AWS error, just print it and continue
                print(f"❌ An AWS ClientError occurred: {e}")
                return "", ""
        except Exception as e:
            print(f"❌ A general error occurred while calling the Bedrock API: {e}")
            return "", ""