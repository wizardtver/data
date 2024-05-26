import requests

if __name__ == '__main__':
    state = int(input('state: '))
    sqft = int(input('size sqft: '))
    total_school = int(input('schools include private: '))
    lng = float(input('longitude: '))
    lat = float(input('latitude: '))
    status = int(input('status object:'))
    pool = int(input('own pool: '))
    home_system = int(input('heating and cooling: '))
    stories = int(input('stories: '))
    propertyType = int(input('type of property: '))
    avg_rating_municipal = float(input('schools rating: '))
    
    r = requests.post('http://localhost:5000/predict', json=[state, sqft, total_school, lng, lat, status, pool, home_system, stories, propertyType, avg_rating_municipal])
    
    print('Status code: {}'.format(r.status_code))
    
    if r.status_code == 200:
        print('Prediction: {}'.format(r.json()['prediction']))
    else:
        print(r.text)