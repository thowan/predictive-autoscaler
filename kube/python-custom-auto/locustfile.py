import random
from locust import HttpUser, task, between

class QuickstartUser(HttpUser):
    wait_time = between(1, 1)

    @task
    def index_page(self):
        self.client.get("/")
    
    def response(self):
        self.close()


    # @task(3)
    # def view_item(self):
    #     item_id = random.randint(1, 10000)
    #     self.client.get(f"/item?id={item_id}", name="/item")

    # def on_start(self):
    #     self.client.post("/login", {"username":"foo", "password":"bar"})