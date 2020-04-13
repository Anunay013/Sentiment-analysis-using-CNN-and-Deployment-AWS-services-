import json
import tweet
import datetime
import time
from tweet import TextPreprocessing
import boto3

sage_maker_client = boto3.client("runtime.sagemaker")
s3_client = boto3.client("s3")

preprocess = TextPreprocessing(max_length_tweet=100, max_length_dictionary=10000)

def lambda_handler(event, context):
    # TODO implement
    tweet = event["tweet"]

    ## Preprocessing
    time_preprocessing_start = time.time() ## Start
    features = preprocess.process_tweet(tweet)
    time_preprocessing_end = time.time() ## End

    model_payload = {
        "embedding_input": features
    }


    ## Model Output

    time_model_start = time.time() ## Start
    response = sage_maker_client.invoke_endpoint(EndpointName='a6-endpoint',\
                                                ContentType='application/json',\
                                                    Body=json.dumps(model_payload))

    time_model_end = time.time() ## End

    result = json.loads(response["Body"].read().decode())

    response = dict()

    if result['predictions'][0][0] >= 0.5:
        response['sentiment'] = "Positive!"
    else:
        response['sentiment'] = "Negative!"

    response['request_time_stamp'] = str(datetime.datetime.now())
    response['tweet'] = tweet
    response['probability'] = result['predictions'][0][0]
    response['preprocess_time'] = time_preprocessing_end - time_preprocessing_start
    response['model_inference_time'] = time_model_end - time_model_start


    s3_client.put_object(Body=json.dumps(response), Bucket='aiops-a6-skand', Key='logs/{}.txt'.format(response['request_time_stamp']))

    # print("Result: {}".format(json.dumps(result, indent=2)))

    return response