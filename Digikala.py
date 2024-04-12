import json
import requests


def getCategoryProducts(productCode, maxPro=50):
    category = json.loads(requests.get(f'https://api.digikala.com/v1/product/{productCode}/').text)['data']['product']['breadcrumb'][-2]['url']['uri'].split('/')[2].replace('category-', '')
    searchPageNum = json.loads(requests.get(f'https://api.digikala.com/v1/categories/{category}/search/').text)['data']['pager']['total_pages']
    categoryProductsCodes = []
    for i in range(1, min(100, searchPageNum) + 1):
        if len(categoryProductsCodes) >= maxPro:
            return categoryProductsCodes[:maxPro]
        products = json.loads(requests.get(f'https://api.digikala.com/v1/categories/{category}/search/?page={i}').text)['data']['products']
        categoryProductsCodes += [product['id'] for product in products]
    return categoryProductsCodes


def getComments(productCodes, maxNum=1000, maxCom=20):
    maxCom = maxNum if len(productCodes) == 1 else maxCom
    productsComments = []
    for productCode in productCodes:
        if len(productsComments) >= maxNum:
            return productsComments[:maxNum]
        productComments = []
        for pageNum in range(1, json.loads(requests.get(f'https://api.digikala.com/v1/product/{productCode}/comments/').text)['data']['pager']['total_pages'] + 1):
            if len(productComments) >= maxCom:
                productComments = productComments[:maxCom]
                break
            for comment in json.loads(requests.get(f'https://api.digikala.com/v1/product/{productCode}/comments/?page={min(100, pageNum)}').text)['data']['comments']:
                productComments.append(comment['body'])
        productsComments += productComments
    return productsComments


if __name__ == '__main__':
    code = 10631965
    codes = getCategoryProducts(code)
    comments = getComments(codes)
    pass

"""from googletrans import Translator

trans_reviews = []

translator = Translator()

for review in reviews:
  trans_reviews.append(translator.translate(review).comments + '. ')

trans_reviews"""
