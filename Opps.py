# inheritance

from abc import ABC, abstractmethod


class Parent:
    def __init__(self):
        self.numer = 10

    def func1(self):
        print("This function is in parent class.")
    
class Child(Parent):
    def __init__(self):
        super().__init__()
        self.numer = 20
    def func2(self):
        print("This function is in child class.")
        print("The value of numer is:", self.numer)

one = Child()
one.func2()

# encapsulation getter and setter methods

class BankAccount:
    def __init__(self, account_number, balance):
        self.__account_number = account_number  # private attribute
        self.__balance = balance  # private attribute
    
    def get_account_number(self):
        return self.__account_number
    
    def get_balance(self):
        return self.__balance
    
    def deposit(self, amount):
        if amount > 0:
            self.__balance += amount
            print(f"Deposited {amount}. New balance: {self.__balance}")
        else:
            print("Deposit amount must be positive.")
        
    def withdraw(self, amount):
        if 0 < amount <= self.__balance:
            self.__balance -= amount
            print(f"Withdrew {amount}. New balance: {self.__balance}")
        else:
            print("Invalid withdrawal amount.")
    
person = BankAccount("123456789", 1000)
print("Account Number:", person.get_account_number())
person.deposit(500)
print("Balance:", person.get_balance())
person.withdraw(200)
print("Balance:", person.get_balance())

# polymorphism

class Animal:
    def speak(self):
        print("Animal speaks")
    
class Dog(Animal):
    def speak(self):
        return "Woof!"
    
class Cat(Animal):
    def speak(self):
        return "Meow!"
    
dog = Dog()
cat = Cat()
print(dog.speak())  # Output: Woof!
print(cat.speak())  # Output: Meow!

# abstraction


class Shape(ABC):
    @abstractmethod
    def area(self):
        pass

    @abstractmethod
    def perimeter(self):
        pass

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

    def perimeter(self):
        return 2 * (self.width + self.height)

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return 3.14 * self.radius ** 2

    def perimeter(self):
        return 2 * 3.14 * self.radius

# Usage
rect = Rectangle(5, 3)
circle = Circle(4)

print("Rectangle Area:", rect.area())
print("Rectangle Perimeter:", rect.perimeter())
print("Circle Area:", circle.area())
print("Circle Perimeter:", circle.perimeter())
