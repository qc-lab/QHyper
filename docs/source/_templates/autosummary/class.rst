{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}


{%- if autotype is defined %}
{%- set objtype = autotype.get(name) or objtype %}
{%- endif %}

.. auto{{ objtype }}:: {{ objname }}
   :show-inheritance:

   {% for item in ['__new__', '__init__'] %}
     {%- if item in members and item not in inherited_members %}
   .. automethod:: {{item}}
     {%- endif %}
   {%- endfor %}

   {%- for item in inherited_members %}
     {%- if item in methods %}
       {%- set dummy = methods.remove(item) %}
     {%- endif %}
     {%- if item in attributes %}
       {%- set dummy = attributes.remove(item) %}
     {%- endif %}
   {%- endfor %}
   {%- for item in ['__new__', '__init__'] %}
     {%- if item in methods %}
       {%- set dummy = methods.remove(item) %}
     {%- endif %}
   {%- endfor %}

   {% block methods_documentation %}
   {%- if methods %}
   .. rubric:: Methods
   {% for item in methods %}
   .. automethod:: {{ item }}
   {%- endfor %}
   {%- endif %}
   {%- endblock %}
{# #}
