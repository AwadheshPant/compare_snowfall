# compare_snowfall
The figure shows radar reflectivity values and snowfall estimates of a flight seg-
ment during the AFLUX campaign. The snowfall rate is derived using a Z-S
relation. The figure also shows plots of the nearest ERA5 time snowfall and
the corresponding box scatter plot for the comparison of values between the
ERA5 values and the resampled flight snowfall estimates. The box-scatter plot
is plotted for ERA5 values for the nearest time to flight segment time. The flight
segement is defined as a flight leg where the the altitude is constant and greater
than 2 km. When these flight segments are selected (ac3airborne python pack-
age), thereafter, the corresponding radar reflectivities values are used to derive
snowfall estimates by using a Z (radar reflectivity) - S (snowfall) relationship.
To compare the radar-derived snowfall estimates with the ERA5 data, resam-
pling is done for the flight samples. For resampling, the flight segment is broken
down into steps of 5 minutes and then a central value (mean) for the step is
calculated. Then, ERA5 point which is closest to this point, spatially, is located
and the neaerest ERA5 time is chosen as a ceiling time, i.e, for any flight time
in between 09:00 to 10:00, the ERA5 time chosen will be 10:00. Please note that
the ERA5 values are present in hourly frequency. The box-scatter plot shows
the variation of flight values plotted on the y-axis with means plotted as green
triangles. The corresponding ERA5 values to these mean values are plotted on
x-axis.

![Example for a segment resampling](image(1).png)
