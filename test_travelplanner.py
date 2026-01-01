import re, string, os, sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "tools/planner")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../tools/planner")))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import importlib
import ast
from typing import List, Dict, Any
import tiktoken
from pandas import DataFrame
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.llms.base import BaseLLM
from langchain.prompts import PromptTemplate
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from utils.func import load_line_json_data, save_file
import sys
import json
import openai
import time
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from langchain_google_genai import ChatGoogleGenerativeAI
import argparse
from datasets import load_dataset
from contextlib import nullcontext
import os
import pdb
from openai_func import *
import json
from z3 import *
from tools.cities.apis import *
from tools.flights.apis import *
from tools.accommodations.apis import *
from tools.attractions.apis import *
from tools.googleDistanceMatrix.apis import *
from tools.restaurants.apis import *
import time

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
# GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']

actionMapping = {"FlightSearch":"flights","AttractionSearch":"attractions","GoogleDistanceMatrix":"googleDistanceMatrix","accommodationSearch":"accommodation","RestaurantSearch":"restaurants","CitySearch":"cities"}


def _safe_literal_eval(value):
    if pd.isna(value):
        return None
    if isinstance(value, (list, dict, tuple)):
        return value
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return None


def _row_to_local_query(row: pd.Series) -> str:
    dests = _safe_literal_eval(row.get('dest')) or []
    if isinstance(dests, str):
        dests = [dests]
    dates = _safe_literal_eval(row.get('date')) or []
    days = int(row.get('days', len(dates)))
    people = int(row.get('people_number', 1))
    visiting_cities = int(row.get('visiting_city_number', max(1, len(dests) or 1)))
    budget = int(float(row.get('budget', 0)))

    dest_phrase = ', '.join(dests) if dests else 'the destination'
    date_phrase = ', '.join(dates) if dates else 'the planned travel dates'
    people_phrase = 'person' if people == 1 else 'people'
    city_phrase = 'city' if visiting_cities == 1 else 'cities'

    query_text = (
        f"Please plan a {days}-day trip for {people} {people_phrase} departing from {row.get('org')} "
        f"to visit {dest_phrase}. The itinerary should cover {visiting_cities} destination {city_phrase} "
        f"between {date_phrase}. The total budget is ${budget}."
    )
    return query_text


def load_local_query_dataset(csv_path: str) -> List[Dict[str, Any]]:
    df = pd.read_csv(csv_path)
    queries: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        queries.append({"query": _row_to_local_query(row)})
    return queries


def convert_to_int(real):
    out = ToInt(real) # ToInt(real + 0.0001)
    out += If(real == out, 0, 1)
    return out

def get_arrivals_list(transportation_arrtime, day, variables):
    arrives = []
    if day == 3: 
        arrives.append(transportation_arrtime[0])
        arrives.append(IntVal(-1))
        arrives.append(transportation_arrtime[1])
    elif day == 5:
        arrives.append(transportation_arrtime[0])
        arrives.append(If(variables[1] == 1, transportation_arrtime[1], IntVal(-1)))
        arrives.append(If(variables[1] == 2, transportation_arrtime[1], IntVal(-1)))
        arrives.append(If(variables[1] == 3, transportation_arrtime[1], IntVal(-1)))
        arrives.append(transportation_arrtime[2])
    else:
        arrives.append(transportation_arrtime[0])
        arrives.append(If(variables[1] == 1, transportation_arrtime[1], If(variables[2] == 1, transportation_arrtime[2], IntVal(-1))))
        arrives.append(If(variables[1] == 2, transportation_arrtime[1], If(variables[2] == 2, transportation_arrtime[2], IntVal(-1))))
        arrives.append(If(variables[1] == 3, transportation_arrtime[1], If(variables[2] == 3, transportation_arrtime[2], IntVal(-1))))
        arrives.append(If(variables[1] == 4, transportation_arrtime[1], If(variables[2] == 4, transportation_arrtime[2], IntVal(-1))))
        arrives.append(If(variables[1] == 5, transportation_arrtime[1], If(variables[2] == 5, transportation_arrtime[2], IntVal(-1))))
        arrives.append(transportation_arrtime[3])
    return arrives

def get_city_list(city, day, departure_dates):
    city_list = []
    if day == 3: 
        city_list.append(IntVal(-1))
        city_list.append(IntVal(0))
        city_list.append(IntVal(0))
        city_list.append(IntVal(-1))
    elif day == 5:
        city_list.append(IntVal(-1))
        city_list.append(city[0])
        city_list.append(If(departure_dates[1] <= 1, city[1],city[0]))
        city_list.append(If(departure_dates[1] <= 2, city[1], city[0]))
        city_list.append(If(departure_dates[1] <= 3, city[1], city[0]))
        city_list.append(IntVal(-1))
    else:
        city_list.append(IntVal(-1))
        city_list.append(city[0])
        city_list.append(If(departure_dates[2] <= 1, city[2],If(departure_dates[1] <= 1, city[1],city[0])))
        city_list.append(If(departure_dates[2] <= 2, city[2], If(departure_dates[1] <= 2, city[1],city[0])))
        city_list.append(If(departure_dates[2] <= 3, city[2], If(departure_dates[1] <= 3, city[1],city[0])))
        city_list.append(If(departure_dates[2] <= 4, city[2], If(departure_dates[1] <= 4, city[1],city[0])))
        city_list.append(If(departure_dates[2] <= 5, city[2], If(departure_dates[1] <= 5, city[1],city[0])))
        city_list.append(IntVal(-1))
    return city_list

def generate_as_plan(s, variables, query):
    CitySearch = Cities()
    FlightSearch = Flights()
    AttractionSearch = Attractions()
    DistanceSearch = GoogleDistanceMatrix()
    AccommodationSearch = Accommodations()
    RestaurantSearch = Restaurants()
    cities = []
    transportation = []
    departure_dates = []
    transportation_info = []
    restaurant_city_list = []
    attraction_city_list = []
    accommodation_city_list = []
    if query['visiting_city_number'] == 1:
        cities = [query['dest']]
        cities_list = [query['dest']]
    else:
        cities_list = CitySearch.run(query['dest'], query['org'], query['date'])
        if query['org'] in cities_list:
            cities_list.remove(query['org'])
        for city in variables['city']:
            cities.append(cities_list[int(s.model()[city].as_long())])
    for i, flight in enumerate(variables['flight']):
        if bool(s.model()[flight]):
            transportation.append('flight')
        elif bool(s.model()[variables['self-driving'][i]]):
            transportation.append('self-driving')
        else:
            transportation.append('taxi')
    for date_index in variables['departure_dates']:
        departure_dates.append(query['date'][int(s.model()[date_index].as_long())])
    dest_cities = [query['org']] + cities + [query['org']]
    for i, index in enumerate(variables['flight_index']):
        if transportation[i] == 'flight':
            flight_index = int(s.model()[index].as_long())
            flight_list = FlightSearch.run(dest_cities[i], dest_cities[i+1], departure_dates[i])
            # flight_info = f'Flight Number: {np.array(flight_list['Flight Number'])[flight_index]}, from {np.array(flight_list['OriginCityName'])[flight_index]} to {np.array(flight_list['DestCityName'])[flight_index]}, Departure Time: {np.array(flight_list['DepTime'])[flight_index]}, Arrival Time: {np.array(flight_list['ArrTime'])[flight_index]}'
            flight_info = 'Flight Number: {}, from {} to {}, Departure Time: {}, Arrival Time: {}'.format(np.array(flight_list['Flight Number'])[flight_index], np.array(flight_list['OriginCityName'])[flight_index], np.array(flight_list['DestCityName'])[flight_index], np.array(flight_list['DepTime'])[flight_index], np.array(flight_list['ArrTime'])[flight_index])
            transportation_info.append(flight_info)
        elif transportation[i] == 'self-driving':
            transportation_info.append('Self-' + DistanceSearch.run(dest_cities[i], dest_cities[i+1], mode='driving'))
        else:
            # pdb.set_trace()
            transportation_info.append(DistanceSearch.run(dest_cities[i], dest_cities[i+1], mode='taxi'))
    for i,which_city in enumerate(variables['restaurant_in_which_city']):
        # pdb.set_trace()
        city_index = int(s.model()[which_city].as_long())
        if city_index == -1:
            restaurant_city_list.append('-')
        else:
            city = cities_list[city_index]
            restaurant_list = RestaurantSearch.run(city)
            restaurant_index = int(s.model()[variables['restaurant_index'][i]].as_long())
            restaurant = np.array(restaurant_list['Name'])[restaurant_index]
            restaurant_city_list.append(restaurant + ', ' + city)

    for i,which_city in enumerate(variables['attraction_in_which_city']):
        city_index = int(s.model()[which_city].as_long())
        if city_index == -1:
            attraction_city_list.append('-')
        else:
            city = cities_list[city_index]
            attraction_list = AttractionSearch.run(city)
            attraction_index = int(s.model()[variables['attraction_index'][i]].as_long())
            attraction = np.array(attraction_list['Name'])[attraction_index]
            attraction_city_list.append(attraction + ', ' + city)

    for i,city in enumerate(cities):
        accommodation_list = AccommodationSearch.run(city)
        accommodation_index = int(s.model()[variables['accommodation_index'][i]].as_long())
        accommodation = np.array(accommodation_list['NAME'])[accommodation_index]
        accommodation_city_list.append(accommodation + ', ' + city)
    print(cities)
    print(transportation)
    print(departure_dates)
    print(transportation_info)
    print(restaurant_city_list)
    print(attraction_city_list)
    print(accommodation_city_list)
    return f'Destination cities: {cities},\nTransportation dates: {departure_dates},\nTransportation methods between cities: {transportation_info},\nRestaurants (3 meals per day): {restaurant_city_list},\nAttractions (1 per day): {attraction_city_list},\nAccommodations (1 per city): {accommodation_city_list}'

def pipeline(query, mode, model, index, model_version = None):
    # ✅ 用 model 代替 gpt_nl，和 main 部分保持一致
    path = f'output/{mode}/{model}/{index}/'
    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(path + 'codes/')
        os.makedirs(path + 'plans/')
    # setup
    with open('prompts/query_to_json.txt', 'r') as file:
        query_to_json_prompt = file.read()
    with open('prompts/constraint_to_step_nl.txt', 'r') as file:
        constraint_to_step_prompt = file.read()
    with open('prompts/step_to_code_destination_cities.txt', 'r') as file:
        step_to_code_destination_cities_prompt = file.read()
    with open('prompts/step_to_code_departure_dates.txt', 'r') as file:
        step_to_code_departure_dates_prompt = file.read()
    with open('prompts/step_to_code_transportation_methods.txt', 'r') as file:
        step_to_code_transportation_methods_prompt = file.read()
    with open('prompts/step_to_code_flight.txt', 'r') as file:
        step_to_code_flight_prompt = file.read()
    with open('prompts/step_to_code_driving.txt', 'r') as file:
        step_to_code_driving_prompt = file.read()
    with open('prompts/step_to_code_restaurant.txt', 'r') as file:
        step_to_code_restaurant_prompt = file.read()
    with open('prompts/step_to_code_attraction.txt', 'r') as file:
        step_to_code_attraction_prompt = file.read()
    with open('prompts/step_to_code_accommodation.txt', 'r') as file:
        step_to_code_accommodation_prompt = file.read()
    with open('prompts/step_to_code_budget.txt', 'r') as file:
        step_to_code_budget_prompt = file.read()
        
    CitySearch = Cities()
    # CitySearch.run('Texas', 'Seattle',["2022-03-10", "2022-03-11", "2022-03-12", "2022-03-13", "2022-03-14", "2022-03-15", "2022-03-16"])
    # pdb.set_trace()
    FlightSearch = Flights()
    AttractionSearch = Attractions()
    DistanceSearch = GoogleDistanceMatrix()
    AccommodationSearch = Accommodations()
    RestaurantSearch = Restaurants()
    s = Optimize()
    variables = {}
    times = []

    step_to_code_prompts = {'Destination cities': step_to_code_destination_cities_prompt, 
                            'Departure dates': step_to_code_departure_dates_prompt,
                            'Transportation methods': step_to_code_transportation_methods_prompt,
                            'Flight information': step_to_code_flight_prompt,
                            'Driving information': step_to_code_driving_prompt,
                            'Restaurant information': step_to_code_restaurant_prompt,
                            'Attraction information': step_to_code_attraction_prompt,
                            'Accommodation information': step_to_code_accommodation_prompt,
                            'Budget': step_to_code_budget_prompt
                            }
    plan = ''
    plan_json = ''
    codes = ''
    success = False
    if model == 'local':
        llm_model_name = model_version or 'local'
    elif model == 'gpt':
        llm_model_name = model_version or 'gpt-4o'
    elif model == 'deepseek-chat':
        llm_model_name = model_version or 'deepseek-chat'
    elif model == 'claude':
        llm_model_name = model_version or 'claude-3-opus-20240229'
    elif model == 'mixtral':
        llm_model_name = model_version or 'mistral-large-latest'
    else:
        raise ValueError(f"Unknown model type: {model}")

    
    try:
        # json generated for postprocess only, not used in inputs to LLMs
        if model in ('gpt', 'local', 'deepseek-chat'):
            query_json = json.loads(GPT_response(query_to_json_prompt + '{' + query + '}\n' + 'JSON:\n', llm_model_name)
                            .replace('```json', '').replace('```', ''))
        elif model == 'claude': query_json = json.loads(Claude_response(query_to_json_prompt + '{' + query + '}\n' + 'JSON:\n').replace('```json', '').replace('```', ''))
        elif model == 'mixtral': query_json = json.loads(Mixtral_response(query_to_json_prompt + '{' + query + '}\n' + 'JSON:\n', 'json').replace('```json', '').replace('```', '')) 
        else: ...
        
        with open(path+'plans/' + 'query.txt', 'w') as f:
            f.write(query)
        f.close()

        with open(path+'plans/' + 'query.json', 'w') as f:
            json.dump(query_json, f)
        f.close()

        print('-----------------query in json format-----------------\n',query_json)
        start = time.time()
        if model in ('gpt', 'local', 'deepseek-chat'): steps = GPT_response(constraint_to_step_prompt + query + '\n' + 'Steps:\n', llm_model_name)
        elif model == 'claude': steps = Claude_response(constraint_to_step_prompt + query + '\n' + 'Steps:\n')
        elif model == 'mixtral': steps = Mixtral_response(constraint_to_step_prompt + query + '\n' + 'Steps:\n')
        else: ...
        json_step = time.time()
        times.append(json_step - start)
        
        with open(path+'plans/' + 'steps.txt', 'w') as f:
            f.write(steps)
        f.close()

        steps = steps.split('\n\n')
        for idx, step in enumerate(steps):
            step_stripped = step.strip()
            if not step_stripped:
                continue

            print(f"======== RAW STEP {idx} ========")
            print(step_stripped)
            print("================================")

            # 按行切分
            lines_step = step_stripped.splitlines()
            # 第一行应该是类似 "# Destination cities #  "
            header_line = lines_step[0].strip()

            # 去掉行首行尾的 #，取中间的标题文字
            # "# Destination cities #  " -> "Destination cities"
            header_clean = header_line.lstrip('#').rstrip('#').strip()

            # 剩下的行作为 body
            body_lines = lines_step[1:]
            body = "\n".join(body_lines).strip()

            print(f"---- Parsed header: {header_clean}")
            print(f"---- Parsed body (first 120 chars): {body[:120]}")

            # 用 header 匹配 step 类型
            prompt = ''
            step_key = ''
            for key in step_to_code_prompts.keys():
                if key in header_clean:
                    print('!!!!!!!!!!KEY!!!!!!!!!!\n', key, '\n')
                    prompt = step_to_code_prompts[key]
                    step_key = key
                    break

            if not prompt:
                raise ValueError(f"Unknown step type. Header: {header_clean}")

            lines = body  # 下面继续沿用原先逻辑

            start = time.time()
            if model in ('gpt', 'local', 'deepseek-chat'):
                code = GPT_response(prompt + lines, llm_model_name)
            elif model == 'claude':
                code = Claude_response(prompt + lines)
            elif model == 'mixtral':
                code = Mixtral_response(
                    prompt + '\nRespond with python codes only, do not add \\ in front of symbols like _ or *.\n'
                    'Follow the indentation of provided examples carefully, indent after for-loops!\n'
                    + lines,
                    'code'
                )
            else:
                ...

            step_code = time.time()
            times.append(step_code - start)
            code = code.replace('```python', '')
            code = code.replace('```', '')
            code = code.replace('\_', '_')
            if step_key != 'Destination cities':
                if query_json['days'] == 3:
                    code = code.replace('\n', '\n    ')
                elif query_json['days'] == 5:
                    code = code.replace('\n', '\n            ')
                else:
                    code = code.replace('\n', '\n                ')
            print('!!!!!!!!!!CODE!!!!!!!!!!\n', code, '\n')
            codes += code + '\n'
            with open(path+'codes/' + f'{step_key}.txt', 'w') as f:
                f.write(code)
            f.close()
        with open('prompts/solve_{}.txt'.format(query_json['days']), 'r') as f:
            codes += f.read()
        with open(path+'codes/' + 'codes.txt', 'w') as f:
            f.write(codes)
        start = time.time()
        exec(codes)
        exec_code = time.time()
        times.append(exec_code - start)
    except Exception as e:
        with open(path+'codes/' + 'codes.txt', 'w') as f:
            f.write(codes)
        f.close()
        with open(path+'plans/' + 'error.txt', 'w') as f:
            f.write(str(e))
            # f.write(e.args)
        f.close()
    with open(path+'plans/' + 'time.txt', 'w') as f:
        for line in times:
            f.write(f"{line}\n")
    
def run_code(mode, user_mode, index):
    path =  f'output/{mode}/{user_mode}/{index}/'
        
    CitySearch = Cities()
    FlightSearch = Flights()
    AttractionSearch = Attractions()
    DistanceSearch = GoogleDistanceMatrix()
    AccommodationSearch = Accommodations()
    RestaurantSearch = Restaurants()
    s = Optimize()
    variables = {}
    success = False

    with open(path+'plans/' + 'query.json', 'r') as f:
      query_json = json.loads(f.read())
    f.close()
    with open(path+'codes/' + 'codes.txt', 'r') as f:
      codes = f.read()
    f.close()
    local_vars = locals()
    start = time.time()
    exec(codes, globals(), local_vars)
    exec_code = time.time()
    print('time', exec_code - start)

if __name__ == '__main__':

    tools_list = ["flights","attractions","accommodations","restaurants","googleDistanceMatrix","cities"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--set_type", type=str, default="validation")
    parser.add_argument("--model_name", type=str, default="gpt") #'gpt', 'claude', 'mixtral', 'local'
    args = parser.parse_args()

    if args.set_type == 'validation':
        print('validation')
        query_data_list = load_dataset('osunlp/TravelPlanner', 'validation')['validation']
    elif args.set_type == 'test':
        print('test')
        query_data_list = load_dataset('osunlp/TravelPlanner', 'test')['test']
    elif args.set_type in ('database_small', 'local_small'):
        print('database_small')
        query_data_list = load_local_query_dataset('database_small/queries/query.csv')
    else:
        query_data_list = load_dataset('osunlp/TravelPlanner', 'train')['train']

    numbers = [i for i in range(1, len(query_data_list) + 1)]
    default_model_version = (
    'gpt-4o'
    if args.model_name == 'gpt' else
    'local'
    if args.model_name == 'local' else
    'deepseek-chat'
    if args.model_name == 'deepseek-chat' else
    'claude-3-opus-20240229'
    if args.model_name == 'claude' else
    None
    )
    callback_ctx = get_openai_callback() if args.model_name == 'gpt' else nullcontext()
    with callback_ctx as cb:
        for number in tqdm(numbers[0:11]):
            path = f'output/{args.set_type}/{args.model_name}/{number}/plans/'
            if not os.path.exists(path + 'plan.txt'):
                print(number)
                query_entry = query_data_list[number - 1]
                query = query_entry['query']
                pipeline(query, args.set_type, args.model_name, number, default_model_version)
